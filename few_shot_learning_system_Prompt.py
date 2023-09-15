import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from meta_neural_network_architectures import VGGReLUNormNetwork
from inner_loop_optimizers import LSLRGradientDescentLearningRule

import prompters


def set_torch_seed(seed):
    """
    Sets the pytorch seeds for current experiment run
    :param seed: The seed (int)
    :return: A random number generator to use
    """
    rng = np.random.RandomState(seed=seed)
    torch_seed = rng.randint(0, 999999)
    torch.manual_seed(seed=torch_seed)

    return rng


class MAMLFewShotClassifier(nn.Module):
    def __init__(self, im_shape, device, args):
        """
        Initializes a MAML few shot learning system
        :param im_shape: The images input size, in batch, c, h, w shape
        :param device: The device to use to use the model on.
        :param args: A namedtuple of arguments specifying various hyperparameters.
        """
        super(MAMLFewShotClassifier, self).__init__()
        self.args = args
        self.device = device
        self.batch_size = args.batch_size
        self.use_cuda = args.use_cuda
        self.im_shape = im_shape
        self.current_epoch = 0

        self.rng = set_torch_seed(seed=args.seed)
        self.classifier = VGGReLUNormNetwork(im_shape=self.im_shape, num_output_classes=self.args.
                                             num_classes_per_set,
                                             args=args, device=device, meta_classifier=True).to(device=self.device)

        # Task-specific learner가 너무 task specific한 지식을 학습하지 못하도록 한다.
        ## 따라서 학습 가능한 Nosie를 support set data에 추가한다
        ## query set에도 추가를 해야할까?
        self.prompter = prompters.padding(args=args, prompt_size=10, image_size=self.im_shape)


        #self.task_learning_rate = args.task_learning_rate

        self.task_learning_rate = args.init_inner_loop_learning_rate

        self.inner_loop_optimizer = LSLRGradientDescentLearningRule(device=device,
                                                                    init_learning_rate=self.task_learning_rate,
                                                                    total_num_inner_loop_steps=self.args.number_of_training_steps_per_iter,
                                                                    use_learnable_learning_rates=self.args.learnable_per_layer_per_step_inner_loop_learning_rate)

        self.inner_loop_optimizer.initialise(
            names_weights_dict=self.get_inner_loop_parameter_dict(params=self.classifier.named_parameters()))



        ## Prompt도 inner-loop에서 학습한다
        self.inner_loop_prompt_optimizer = LSLRGradientDescentLearningRule(device=device,
                                                                    init_learning_rate=self.task_learning_rate,
                                                                    total_num_inner_loop_steps=self.args.number_of_training_steps_per_iter,
                                                                    use_learnable_learning_rates=self.args.learnable_per_layer_per_step_inner_loop_learning_rate)

        self.inner_loop_prompt_optimizer.initialise(
            names_weights_dict=self.get_inner_loop_parameter_dict(params=self.prompter.named_parameters()))




        print("Inner Loop parameters")
        for key, value in self.inner_loop_optimizer.named_parameters():
            print(key, value.shape)

        for key, value in self.inner_loop_prompt_optimizer.named_parameters():
            print(key, value.shape)

        self.use_cuda = args.use_cuda
        self.device = device
        self.args = args
        self.to(device)
        print("Outer Loop parameters")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.shape, param.device, param.requires_grad)


        self.optimizer = optim.Adam(self.trainable_parameters(), lr=args.meta_learning_rate, amsgrad=False)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.args.total_epochs,
                                                              eta_min=self.args.min_learning_rate)

        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.to(torch.cuda.current_device())
                self.classifier = nn.DataParallel(module=self.classifier)
            else:
                self.to(torch.cuda.current_device())

            self.device = torch.cuda.current_device()

    def get_per_step_loss_importance_vector(self):
        """
        Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
        loss towards the optimization loss.
        :return: A tensor to be used to compute the weighted average of the loss, useful for
        the MSL (Multi Step Loss) mechanism.
        """
        loss_weights = np.ones(shape=(self.args.number_of_training_steps_per_iter)) * (
                1.0 / self.args.number_of_training_steps_per_iter)
        decay_rate = 1.0 / self.args.number_of_training_steps_per_iter / self.args.multi_step_loss_num_epochs
        min_value_for_non_final_losses = 0.03 / self.args.number_of_training_steps_per_iter
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (self.current_epoch * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1] + (self.current_epoch * (self.args.number_of_training_steps_per_iter - 1) * decay_rate),
            1.0 - ((self.args.number_of_training_steps_per_iter - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).to(device=self.device)
        return loss_weights

    def get_inner_loop_parameter_dict(self, params):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        param_dict = dict()
        for name, param in params:
            if param.requires_grad:
                if self.args.enable_inner_loop_optimizable_bn_params:
                    param_dict[name] = param.to(device=self.device)
                else:
                    if "norm_layer" not in name:
                        param_dict[name] = param.to(device=self.device)

        return param_dict

    def setGrad_inner_loop_parameter_dict(self, params):

        param_dict = dict()
        for name, param in params:
            if "norm_layer" not in name:
                param_dict[name] = param.detach().to(device=self.device)
                param_dict[name].requires_grad = False

        return param_dict

    def apply_inner_loop_update(self, loss, names_weights_copy, inner_loop_optimizer, use_second_order, current_step_idx):

        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.classifier.module.zero_grad(params=names_weights_copy)
        else:
            self.classifier.zero_grad(params=names_weights_copy)
            self.prompter.zero_grad(params=names_weights_copy)

        # 가중치 업데이트 확인용 변수
        # prev_weights = {}
        # for name, param in names_weights_copy.items():
        #     prev_weights[name] = param.data.clone()

        grads = torch.autograd.grad(loss, names_weights_copy.values(),
                                    create_graph=use_second_order, allow_unused=True)
        names_grads_copy = dict(zip(names_weights_copy.keys(), grads))

        names_weights_copy = {key: value[0] for key, value in names_weights_copy.items()}

        for key, grad in names_grads_copy.items():
            if grad is None:
                print('Grads not found for inner loop parameter', key)
            names_grads_copy[key] = names_grads_copy[key].sum(dim=0)

        names_weights_copy = inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy,
                                                                     names_grads_wrt_params_dict=names_grads_copy,
                                                                     num_step=current_step_idx)

        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        names_weights_copy = {
            name.replace('module.', ''): value.unsqueeze(0).repeat(
                [num_devices] + [1 for i in range(len(value.shape))]) for
            name, value in names_weights_copy.items()}

        ## 가중치 업데이트 확인
        # for name, param in names_weights_copy.items():
        #     if not torch.equal(prev_weights[name], param.data):
        #         print(f"{name} inner-loop 가중치가 업데이트되었습니다.")
        #         prev_weights[name] = param.data.clone()

        return names_weights_copy

    def get_across_task_loss_metrics(self, total_losses, total_accuracies):
        losses = {'loss': torch.mean(torch.stack(total_losses))}

        losses['accuracy'] = np.mean(total_accuracies)

        return losses

    def forward(self, data_batch, epoch, use_second_order, use_multi_step_loss_optimization, num_steps, training_phase):
        """
        Runs a forward outer loop pass on the batch of tasks using the MAML/++ framework.
        :param data_batch: A data batch containing the support and target sets.
        :param epoch: Current epoch's index
        :param use_second_order: A boolean saying whether to use second order derivatives.
        :param use_multi_step_loss_optimization: Whether to optimize on the outer loop using just the last step's
        target loss (True) or whether to use multi step loss which improves the stability of the system (False)
        :param num_steps: Number of inner loop steps.
        :param training_phase: Whether this is a training phase (True) or an evaluation phase (False)
        :return: A dictionary with the collected losses of the current outer forward propagation.
        """
        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        [b, ncs, spc] = y_support_set.shape

        self.num_classes_per_set = ncs

        total_classifier_losses = []
        total_prompter_losses = []

        total_accuracies = []
        per_task_target_preds = [[] for i in range(len(x_target_set))]

        self.classifier.zero_grad()
        self.prompter.zero_grad()

        task_accuracies = []
        for task_id, (x_support_set_task, y_support_set_task, x_target_set_task, y_target_set_task) in enumerate(zip(x_support_set,
                              y_support_set,
                              x_target_set,
                              y_target_set)):

            task_classifier_losses = []
            task_prompter_losses = []
            per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector()


            # gradient update가 가능한 weight (requires_grad = True)
            names_weights_copy = self.get_inner_loop_parameter_dict(self.classifier.named_parameters())
            names_prompt_weights_copy = self.get_inner_loop_parameter_dict(self.prompter.named_parameters())

            # gradient update가 가능한 weight (requires_grad = False)
            names_weights_copy_notUpdated = self.setGrad_inner_loop_parameter_dict(self.classifier.named_parameters())
            names_prompt_weights_copy_notUpdated = self.setGrad_inner_loop_parameter_dict(self.prompter.named_parameters())

            num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

            names_weights_copy = {
                name.replace('module.', ''): value.unsqueeze(0).repeat(
                    [num_devices] + [1 for i in range(len(value.shape))]) for
                name, value in names_weights_copy.items()}

            names_prompt_weights_copy = {
                name.replace('module.', ''): value.unsqueeze(0).repeat(
                    [num_devices] + [1 for i in range(len(value.shape))]) for
                name, value in names_prompt_weights_copy.items()}

            names_weights_copy_notUpdated = {
                name.replace('module.', ''): value.unsqueeze(0).repeat(
                    [num_devices] + [1 for i in range(len(value.shape))]) for
                name, value in names_weights_copy_notUpdated.items()}

            names_prompt_weights_copy_notUpdated = {
                name.replace('module.', ''): value.unsqueeze(0).repeat(
                    [num_devices] + [1 for i in range(len(value.shape))]) for
                name, value in names_prompt_weights_copy_notUpdated.items()}


            n, s, c, h, w = x_target_set_task.shape

            x_support_set_task = x_support_set_task.view(-1, c, h, w)
            y_support_set_task = y_support_set_task.view(-1)
            x_target_set_task = x_target_set_task.view(-1, c, h, w)
            y_target_set_task = y_target_set_task.view(-1)

            for num_step in range(num_steps):

                # alternative optimization
                support_prompt_loss, support_classifer_loss, support_preds = self.net_forward(
                    x=x_support_set_task,
                    y=y_support_set_task,
                    prompt_weights=names_prompt_weights_copy,
                    prompt_weights_notUpdated=names_prompt_weights_copy_notUpdated,
                    classifier_weights=names_weights_copy,
                    classifier_weights_notUpdated=names_weights_copy_notUpdated,
                    backup_running_statistics=num_step == 0,
                    training=True,
                    num_step=num_step)

                # Prompt의 paramter를 inner-loop에서 update한다
                names_prompt_weights_copy = self.apply_inner_loop_update(loss=support_prompt_loss,
                                                                         names_weights_copy=names_prompt_weights_copy,
                                                                         inner_loop_optimizer=self.inner_loop_prompt_optimizer,
                                                                         use_second_order=use_second_order,
                                                                         current_step_idx=num_step)
                # Classifier의 parameter를 inner-loop에서 update한다
                names_weights_copy = self.apply_inner_loop_update(loss=support_classifer_loss,
                                                                  names_weights_copy=names_weights_copy,
                                                                  inner_loop_optimizer=self.inner_loop_optimizer,
                                                                  use_second_order=use_second_order,
                                                                  current_step_idx=num_step)

                if use_multi_step_loss_optimization and training_phase and epoch < self.args.multi_step_loss_num_epochs:
                    target_prompt_loss, target_classifier_loss, target_preds = self.net_forward(x=x_target_set_task,
                                                                                y=y_target_set_task,
                                                                               prompt_weights=names_prompt_weights_copy,
                                                                               prompt_weights_notUpdated=names_prompt_weights_copy_notUpdated,
                                                                                classifier_weights=names_weights_copy,
                                                                                classifier_weights_notUpdated=names_weights_copy_notUpdated,
                                                                                backup_running_statistics=False, training=True,
                                                                                num_step=num_step,
                                                                                isQueryset=True)
                    task_classifier_losses.append(per_step_loss_importance_vectors[num_step] * target_classifier_loss)

                elif num_step == (self.args.number_of_training_steps_per_iter - 1):
                    target_prompt_loss, target_classifier_loss, target_preds = self.net_forward(x=x_target_set_task,
                                                                                y=y_target_set_task,
                                                                               prompt_weights=names_prompt_weights_copy,
                                                                               prompt_weights_notUpdated=names_prompt_weights_copy_notUpdated,
                                                                                classifier_weights=names_weights_copy,
                                                                                classifier_weights_notUpdated=names_weights_copy_notUpdated,
                                                                                backup_running_statistics=False, training=True,
                                                                                num_step=num_step,
                                                                                isQueryset=True)
                    task_prompter_losses.append(target_prompt_loss)
                    task_classifier_losses.append(target_classifier_loss)

            per_task_target_preds[task_id] = target_preds.detach().cpu().numpy()
            _, predicted = torch.max(target_preds.data, 1)

            accuracy = predicted.float().eq(y_target_set_task.data.float()).cpu().float()

            # prompter loss 저장
            task_prompter_losses = torch.sum(torch.stack(task_prompter_losses))
            total_prompter_losses.append(task_prompter_losses)

            # classifer loss 저장
            task_classifier_losses = torch.sum(torch.stack(task_classifier_losses))
            total_classifier_losses.append(task_classifier_losses)

            total_accuracies.extend(accuracy)

            if not training_phase:
                self.classifier.restore_backup_stats()

        total_classifier_losses = self.get_across_task_loss_metrics(total_losses=total_classifier_losses,
                                                   total_accuracies=total_accuracies)

        total_prompter_losses = self.get_across_task_loss_metrics(total_losses=total_prompter_losses,
                                                                    total_accuracies=total_accuracies)

        for idx, item in enumerate(per_step_loss_importance_vectors):
            total_classifier_losses['loss_importance_vector_{}'.format(idx)] = item.detach().cpu().numpy()

        return total_classifier_losses, total_prompter_losses, per_task_target_preds

    def net_forward(self, x, y, prompt_weights, classifier_weights, prompt_weights_notUpdated, classifier_weights_notUpdated, backup_running_statistics, training, num_step, isQueryset=False):

        ## requires_grad check
        for name, param in prompt_weights.items():
            if not param.requires_grad:
                print("prompt_weights == ", name)

        for name, param in classifier_weights.items():
            if not param.requires_grad:
                print("classifier_weights == ", name)

        for name, param in prompt_weights_notUpdated.items():
            if param.requires_grad:
                print("prompt_weights == ", name)

        for name, param in classifier_weights_notUpdated.items():
            if param.requires_grad:
                print("classifier_weights_notUpdated == ", name)
        #####

        if not isQueryset:
            # Prompter의 weight를 update하기 위한 로직
            prompted_images_x = self.prompter.forward(x=x, params=prompt_weights)
            preds1 = self.classifier.forward(x=prompted_images_x, params=classifier_weights_notUpdated,
                                            training=training,
                                            backup_running_statistics=backup_running_statistics, num_step=num_step)
            prompt_loss = F.cross_entropy(input=preds1, target=y)

            # Classifier의 weight를 update하기 위한 로직
            prompted_images_x2 = self.prompter.forward(x, prompt_weights_notUpdated)
            preds = self.classifier.forward(x=prompted_images_x2, params=classifier_weights,
                                            training=training,
                                            backup_running_statistics=backup_running_statistics, num_step=num_step)
            classifier_loss = F.cross_entropy(input=preds, target=y)
        else:
            # Queryset인 경우 prompt를 추가하지 않는다.
            preds = self.classifier.forward(x=x, params=classifier_weights,
                                            training=training,
                                            backup_running_statistics=backup_running_statistics, num_step=num_step)

            classifier_loss = F.cross_entropy(input=preds, target=y)
            prompt_loss = classifier_loss

        return prompt_loss, classifier_loss, preds

    def trainable_parameters(self):
        """
        Returns an iterator over the trainable parameters of the model.
        """
        for param in self.parameters():
            if param.requires_grad:
                yield param

    def train_forward_prop(self, data_batch, epoch):
        """
        Runs an outer loop forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        total_classifier_losses, total_prompter_losses, per_task_target_preds = self.forward(data_batch=data_batch, epoch=epoch,
                                                     use_second_order=self.args.second_order and
                                                                      epoch > self.args.first_order_to_second_order_epoch,
                                                     use_multi_step_loss_optimization=self.args.use_multi_step_loss_optimization,
                                                     num_steps=self.args.number_of_training_steps_per_iter,
                                                     training_phase=True)

        return total_classifier_losses, total_prompter_losses, per_task_target_preds

    def evaluation_forward_prop(self, data_batch, epoch):
        """
        Runs an outer loop evaluation forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        total_classifier_losses, total_prompter_losses, per_task_target_preds = self.forward(data_batch=data_batch, epoch=epoch, use_second_order=False,
                                                     use_multi_step_loss_optimization=True,
                                                     num_steps=self.args.number_of_evaluation_steps_per_iter,
                                                     training_phase=False)

        return total_classifier_losses, total_prompter_losses, per_task_target_preds

    def meta_update(self, total_classifier_losses, total_prompter_losses):
        """
        Applies an outer loop update on the meta-parameters of the model.
        :param loss: The current crossentropy loss.
        """

        # ## 가중치 업데이트 확인용 변수
        # prev_weights = {}
        # for name, param in self.prompter.named_parameters():
        #     prev_weights[name] = param.data.clone()

        self.optimizer.zero_grad()

        total_classifier_losses.backward()

        #total_prompter_losses.backward()

        # if 'imagenet' in self.args.dataset_name:
        #     for name, param in self.classifier.named_parameters():
        #         if param.requires_grad:
        #             param.grad.data.clamp_(-10, 10)  # not sure if this is necessary, more experiments are needed

        self.optimizer.step()

        # ## 가중치 업데이트 확인
        # for name, param in self.prompter.named_parameters():
        #     if not torch.equal(prev_weights[name], param.data):
        #         print(f"{name} 가중치가 업데이트되었습니다.")
        #         prev_weights[name] = param.data.clone()



    def run_train_iter(self, data_batch, epoch):
        """
        Runs an outer loop update step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """
        epoch = int(epoch)
        self.scheduler.step(epoch=epoch)
        if self.current_epoch != epoch:
            self.current_epoch = epoch

        if not self.training:
            self.train()

        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        x_support_set = torch.Tensor(x_support_set).float().to(device=self.device)
        x_target_set = torch.Tensor(x_target_set).float().to(device=self.device)
        y_support_set = torch.Tensor(y_support_set).long().to(device=self.device)
        y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)

        data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        total_classifier_losses, total_prompter_losses, per_task_target_preds = self.train_forward_prop(data_batch=data_batch, epoch=epoch)

        self.meta_update(total_classifier_losses=total_classifier_losses['loss'], total_prompter_losses=total_prompter_losses['loss'])
        total_classifier_losses['learning_rate'] = self.scheduler.get_lr()[0]
        self.optimizer.zero_grad()
        self.zero_grad()

        return total_classifier_losses, per_task_target_preds

    def run_validation_iter(self, data_batch):
        """
        Runs an outer loop evaluation step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """

        if self.training:
            self.eval()

        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        x_support_set = torch.Tensor(x_support_set).float().to(device=self.device)
        x_target_set = torch.Tensor(x_target_set).float().to(device=self.device)
        y_support_set = torch.Tensor(y_support_set).long().to(device=self.device)
        y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)

        data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        total_classifier_losses, total_prompter_losses, per_task_target_preds = self.evaluation_forward_prop(data_batch=data_batch, epoch=self.current_epoch)
        losses = total_classifier_losses
        # losses['loss'].backward() # uncomment if you get the weird memory error
        # self.zero_grad()
        # self.optimizer.zero_grad()

        return losses, per_task_target_preds

    def save_model(self, model_save_dir, state):
        """
        Save the network parameter state and experiment state dictionary.
        :param model_save_dir: The directory to store the state at.
        :param state: The state containing the experiment state and the network. It's in the form of a dictionary
        object.
        """
        state['network'] = self.state_dict()
        state['optimizer'] = self.optimizer.state_dict()
        torch.save(state, f=model_save_dir)

    def load_model(self, model_save_dir, model_name, model_idx):
        """
        Load checkpoint and return the state dictionary containing the network state params and experiment state.
        :param model_save_dir: The directory from which to load the files.
        :param model_name: The model_name to be loaded from the direcotry.
        :param model_idx: The index of the model (i.e. epoch number or 'latest' for the latest saved model of the current
        experiment)
        :return: A dictionary containing the experiment state and the saved model parameters.
        """
        filepath = os.path.join(model_save_dir, "{}_{}".format(model_name, model_idx))
        state = torch.load(filepath)
        state_dict_loaded = state['network']
        self.optimizer.load_state_dict(state['optimizer'])
        self.load_state_dict(state_dict=state_dict_loaded)
        return state