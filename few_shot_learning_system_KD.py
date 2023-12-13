import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from meta_neural_network_architectures import VGGReLUNormNetwork,ResNet12
from inner_loop_optimizers_GR import GradientDescentLearningRule, LSLRGradientDescentLearningRule

from SAM import SAM

from timm.loss import LabelSmoothingCrossEntropy
from loss import knowledge_distillation_loss


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

        if self.args.backbone == 'ResNet12':
            self.classifier = ResNet12(im_shape=self.im_shape, num_output_classes=self.args.
                                       num_classes_per_set,
                                       args=args, device=device, meta_classifier=True).to(device=self.device)
        else:
            self.classifier = VGGReLUNormNetwork(im_shape=self.im_shape, num_output_classes=self.args.
                                                 num_classes_per_set,
                                                 args=args, device=device, meta_classifier=True).to(device=self.device)

        #self.task_learning_rate = args.task_learning_rate

        self.task_learning_rate = args.init_inner_loop_learning_rate

        names_weights_copy = self.get_inner_loop_parameter_dict(self.classifier.named_parameters())

        if self.args.learnable_per_layer_per_step_inner_loop_learning_rate:
            self.inner_loop_optimizer = LSLRGradientDescentLearningRule(device=device,
                                                                        args=self.args,
                                                                        init_learning_rate=self.task_learning_rate,
                                                                        total_num_inner_loop_steps=self.args.number_of_training_steps_per_iter,
                                                                        use_learnable_learning_rates=True)
            self.inner_loop_optimizer.initialise(names_weights_dict=names_weights_copy)
        else:
            self.inner_loop_optimizer = GradientDescentLearningRule(device=device,
                                                                    args=self.args,
                                                                    learning_rate=self.task_learning_rate)

        # Gradient Arbiter
        if self.args.arbiter:
            num_layers = len(names_weights_copy)
            input_dim = num_layers * 2
            output_dim = num_layers
            self.arbiter = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(inplace=True),
                nn.Linear(input_dim, output_dim),
                ## nn.Softplus(beta=2) # GAP
                nn.Softplus() # CxGrad
            ).to(device=self.device)

        print("Inner Loop parameters")
        for key, value in self.inner_loop_optimizer.named_parameters():
            print(key, value.shape)

        self.use_cuda = args.use_cuda
        self.device = device
        self.args = args
        self.to(device)
        print("Outer Loop parameters")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.shape, param.device, param.requires_grad)

        base_optimizer = optim.Adam
        # base_optimizer = optim.SGD
        # optim.Adam(self.trainable_parameters(), lr=args.meta_learning_rate, amsgrad=False)

        self.optimizer = SAM(self.trainable_parameters(), base_optimizer,
                             adaptive=False, lr=args.meta_learning_rate)

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

    def apply_inner_loop_update(self, loss, names_weights_copy, out_feature_dict, alpha, use_second_order, current_step_idx, current_iter, training_phase):
        """
        Applies an inner loop update given current step's loss, the weights to update, a flag indicating whether to use
        second order derivatives and the current step's index.
        :param loss: Current step's loss with respect to the support set.
        :param names_weights_copy: A dictionary with names to parameters to update.
        :param use_second_order: A boolean flag of whether to use second order derivatives.
        :param current_step_idx: Current step's index.
        :return: A dictionary with the updated weights (name, param)
        """
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.classifier.module.zero_grad(params=names_weights_copy)
        else:
            self.classifier.zero_grad(params=names_weights_copy)

        grads = torch.autograd.grad(loss, names_weights_copy.values(),
                                    create_graph=use_second_order, allow_unused=True)


        names_grads_copy = dict(zip(names_weights_copy.keys(), grads))

        names_weights_copy = {key: value[0] for key, value in names_weights_copy.items()}

        for key, grad in names_grads_copy.items():
            if grad is None:
                print('Grads not found for inner loop parameter', key)
            names_grads_copy[key] = names_grads_copy[key].sum(dim=0)


        names_weights_copy = self.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy,
                                                                     names_grads_wrt_params_dict=names_grads_copy,
                                                                     out_feature_dict=out_feature_dict,
                                                                     generated_alpha_params=alpha,
                                                                     num_step=current_step_idx,
                                                                     current_iter=current_iter,
                                                                     training_phase=training_phase)

        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        names_weights_copy = {
            name.replace('module.', ''): value.unsqueeze(0).repeat(
                [num_devices] + [1 for i in range(len(value.shape))]) for
            name, value in names_weights_copy.items()}


        return names_weights_copy

    def get_across_task_loss_metrics(self, total_losses, total_accuracies, task_gradients):

        losses = dict()

        task1_gradient = task_gradients[0]['layer_dict.conv3.conv.weight']

        task2_gradient = task_gradients[1]['layer_dict.conv3.conv.weight']

        # 각 텐서를 벡터로 평탄화(flatten)
        task1_gradient = task1_gradient.view(task1_gradient.size(0), -1)
        task2_gradient = task2_gradient.view(task2_gradient.size(0), -1)

        cosine_similarity = torch.abs(F.cosine_similarity(task1_gradient, task2_gradient))

        losses['loss'] = torch.mean(torch.stack(total_losses)) + cosine_similarity
        losses['accuracy'] = np.mean(total_accuracies)

        return losses

    def get_soft_label(self, x_support_set_task, y_support_set_task, x_target_set_task, y_target_set_task, names_weights_copy, epoch):

        support_loss, support_preds, out_feature_dict = self.net_forward(
            x=x_support_set_task,
            y=y_support_set_task,
            weights=names_weights_copy,
            backup_running_statistics=True,
            training=True,
            num_step=0,
            epoch=epoch
        )

        taget_loss, target_preds, out_feature_dict = self.net_forward(
            x=x_target_set_task,
            y=y_target_set_task,
            weights=names_weights_copy,
            backup_running_statistics=True,
            training=True,
            num_step=0,
            epoch=epoch
        )

        return support_preds.detach(), target_preds.detach() # detach하여 역전파 방지



    def forward(self, data_batch, epoch, use_second_order, use_multi_step_loss_optimization, num_steps, training_phase, current_iter):
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

        task_gradient = []

        total_losses = []
        total_accuracies = []
        per_task_target_preds = [[] for i in range(len(x_target_set))]
        self.classifier.zero_grad()
        task_accuracies = []
        for task_id, (x_support_set_task, y_support_set_task, x_target_set_task, y_target_set_task) in enumerate(zip(x_support_set,
                              y_support_set,
                              x_target_set,
                              y_target_set)):
            task_losses = []
            per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector()
            names_weights_copy = self.get_inner_loop_parameter_dict(self.classifier.named_parameters())

            num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

            names_weights_copy = {
                name.replace('module.', ''): value.unsqueeze(0).repeat(
                    [num_devices] + [1 for i in range(len(value.shape))]) for
                name, value in names_weights_copy.items()}

            n, s, c, h, w = x_target_set_task.shape

            x_support_set_task = x_support_set_task.view(-1, c, h, w)
            y_support_set_task = y_support_set_task.view(-1)
            x_target_set_task = x_target_set_task.view(-1, c, h, w)
            y_target_set_task = y_target_set_task.view(-1)

            support_soft_preds, targetsoft__preds = self.get_soft_label(x_support_set_task, y_support_set_task, x_target_set_task, y_target_set_task, names_weights_copy, epoch)

            for num_step in range(num_steps):

                support_loss, support_preds, out_feature_dict  = self.net_forward(
                    x=x_support_set_task,
                    y=y_support_set_task,
                    weights=names_weights_copy,
                    backup_running_statistics=num_step == 0,
                    training=True,
                    num_step=num_step,
                    soft_target=support_soft_preds,
                    epoch=epoch
                )

                generated_alpha_params = {}

                if self.args.arbiter:
                    support_loss_grad = torch.autograd.grad(support_loss, names_weights_copy.values(),
                                                            retain_graph=True)

                    names_grads_copy = dict(zip(names_weights_copy.keys(), support_loss_grad))

                    per_step_task_embedding = []

                    for key, weight in names_weights_copy.items():
                        weight_norm = torch.norm(weight, p=2)
                        per_step_task_embedding.append(weight_norm)

                    for key, grad in names_grads_copy.items():
                        gradient_l2norm = torch.norm(grad, p=2)
                        per_step_task_embedding.append(gradient_l2norm)

                    per_step_task_embedding = torch.stack(per_step_task_embedding)

                    ## Standardization
                    per_step_task_embedding = (per_step_task_embedding - per_step_task_embedding.mean()) / (
                                per_step_task_embedding.std() + 1e-12)

                    generated_gradient_rate = self.arbiter(per_step_task_embedding)

                    g = 0
                    for key in names_weights_copy.keys():
                        generated_alpha_params[key] = generated_gradient_rate[g]
                        g += 1

                names_weights_copy = self.apply_inner_loop_update(loss=support_loss,
                                                                  names_weights_copy=names_weights_copy,
                                                                  out_feature_dict=out_feature_dict,
                                                                  alpha=generated_alpha_params,
                                                                  use_second_order=use_second_order,
                                                                  current_step_idx=num_step,
                                                                  current_iter=current_iter,
                                                                  training_phase=training_phase)

                if use_multi_step_loss_optimization and training_phase and epoch < self.args.multi_step_loss_num_epochs:
                    target_loss, target_preds, _ = self.net_forward(x=x_target_set_task,
                                                                 y=y_target_set_task, weights=names_weights_copy,
                                                                 backup_running_statistics=False, training=True,
                                                                 num_step=num_step)

                    task_losses.append(per_step_loss_importance_vectors[num_step] * target_loss)
                elif num_step == (self.args.number_of_training_steps_per_iter - 1):
                    target_loss, target_preds, _ = self.net_forward(x=x_target_set_task,
                                                                 y=y_target_set_task, weights=names_weights_copy,
                                                                 backup_running_statistics=False, training=True,
                                                                 num_step=num_step,soft_target=targetsoft__preds, epoch=epoch)


                    # names_weights_copy 때문에 무조건 여기서 계산을 해야한다
                    target_loss_grad = torch.autograd.grad(target_loss, names_weights_copy.values(), retain_graph=True)
                    target_grads_copy = dict(zip(names_weights_copy.keys(), target_loss_grad))

                    task_gradient.append(target_grads_copy)

                    task_losses.append(target_loss)
            ## Inner-loop END

            per_task_target_preds[task_id] = target_preds.detach().cpu().numpy()
            _, predicted = torch.max(target_preds.data, 1)

            accuracy = predicted.float().eq(y_target_set_task.data.float()).cpu().float()

            task_losses = torch.sum(torch.stack(task_losses))

            total_losses.append(task_losses)
            total_accuracies.extend(accuracy)

        ## Outer-loop End

            if not training_phase:
                self.classifier.restore_backup_stats()

        losses = self.get_across_task_loss_metrics(total_losses=total_losses,
                                                   total_accuracies=total_accuracies,
                                                   task_gradients=task_gradient)

        for idx, item in enumerate(per_step_loss_importance_vectors):
            losses['loss_importance_vector_{}'.format(idx)] = item.detach().cpu().numpy()

        return losses, per_task_target_preds

    def net_forward(self, x, y, weights, backup_running_statistics, training, num_step, epoch, soft_target=None):
        """
        A base model forward pass on some data points x. Using the parameters in the weights dictionary. Also requires
        boolean flags indicating whether to reset the running statistics at the end of the run (if at evaluation phase).
        A flag indicating whether this is the training session and an int indicating the current step's number in the
        inner loop.
        :param x: A data batch of shape b, c, h, w
        :param y: A data targets batch of shape b, n_classes
        :param weights: A dictionary containing the weights to pass to the network.
        :param backup_running_statistics: A flag indicating whether to reset the batch norm running statistics to their
         previous values after the run (only for evaluation)
        :param training: A flag indicating whether the current process phase is a training or evaluation.
        :param num_step: An integer indicating the number of the step in the inner loop.
        :return: the crossentropy losses with respect to the given y, the predictions of the base model.
        """
        preds, out_feature_dict = self.classifier.forward(x=x, params=weights,
                                        training=training,
                                        backup_running_statistics=backup_running_statistics, num_step=num_step)

        if self.args.smoothing:

            if soft_target == None:
                criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
                loss = criterion(preds, y)
            else:
                # alpha = epoch / self.args.total_epochs
                alpha = 0.1
                loss = knowledge_distillation_loss(outputs_student=preds, outputs_teacher=soft_target, labels=y, alpha=alpha)

        else:
            loss = F.cross_entropy(input=preds, target=y)

        return loss, preds, out_feature_dict

    def trainable_parameters(self):
        """
        Returns an iterator over the trainable parameters of the model.
        """
        for param in self.parameters():
            if param.requires_grad:
                yield param

    def train_forward_prop(self, data_batch, epoch, current_iter):
        """
        Runs an outer loop forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses, per_task_target_preds = self.forward(data_batch=data_batch, epoch=epoch,
                                                     use_second_order=self.args.second_order and
                                                                      epoch > self.args.first_order_to_second_order_epoch,
                                                     use_multi_step_loss_optimization=self.args.use_multi_step_loss_optimization,
                                                     num_steps=self.args.number_of_training_steps_per_iter,
                                                     training_phase=True,
                                                     current_iter=current_iter)
        return losses, per_task_target_preds

    def evaluation_forward_prop(self, data_batch, epoch, current_iter):
        """
        Runs an outer loop evaluation forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses, per_task_target_preds = self.forward(data_batch=data_batch, epoch=epoch, use_second_order=False,
                                                     use_multi_step_loss_optimization=True,
                                                     num_steps=self.args.number_of_evaluation_steps_per_iter,
                                                     training_phase=False,
                                                     current_iter=current_iter)

        return losses, per_task_target_preds

    def meta_update(self, loss, current_iter, first_step):
        """
        Applies an outer loop update on the meta-parameters of the model.
        :param loss: The current crossentropy loss.
        """

        # 가중치 업데이트 확인용 변수
        # prev_weights = {}
        # for name, param in self.arbiter.named_parameters():
        #     prev_weights[name] = param.data.clone()

        loss.backward()

        if first_step:
            self.optimizer.first_step(zero_grad=True)
        else:
            self.optimizer.second_step(zero_grad=True)

        # if 'imagenet' in self.args.dataset_name:
        #     for name, param in self.classifier.named_parameters():
        #         if param.requires_grad:
        #             param.grad.data.clamp_(-10, 10)  # not sure if this is necessary, more experiments are needed

        # if self.args.arbiter:
        #     # Outer-loop에서도 Gradient norm
        #     for name, param in self.classifier.named_parameters():
        #         if param.requires_grad:
        #             param.grad = param.grad / torch.norm(param.grad, p=2)

        #self.optimizer.step()

        # 가중치 업데이트 확인
        # for name, param in self.arbiter.named_parameters():
        #     if not torch.equal(prev_weights[name], param.data):
        #         print(f"{name} 가중치가 업데이트되었습니다.")
        #         prev_weights[name] = param.data.clone()

    def run_train_iter(self, data_batch, epoch, current_iter):
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

        losses_1, per_task_target_preds_1 = self.train_forward_prop(data_batch=data_batch, epoch=epoch, current_iter=current_iter)

        self.optimizer.zero_grad()

        self.meta_update(loss=losses_1['loss'], current_iter=current_iter, first_step=True)

        losses, per_task_target_preds = self.train_forward_prop(data_batch=data_batch, epoch=epoch, current_iter=current_iter)

        self.meta_update(loss=losses['loss'], current_iter=current_iter, first_step=False)


        losses['learning_rate'] = self.scheduler.get_lr()[0]

        self.optimizer.zero_grad()
        self.zero_grad()

        return losses, per_task_target_preds

    def run_validation_iter(self, data_batch, current_iter):
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

        losses, per_task_target_preds = self.evaluation_forward_prop(data_batch=data_batch, epoch=self.current_epoch, current_iter=current_iter)

        losses['loss'].backward() # uncomment if you get the weird memory error
        self.zero_grad()
        self.optimizer.zero_grad()

        return losses, per_task_target_preds

    def save_model(self, model_save_dir, state):
        """
        Save the network parameter state and experiment state dictionary.
        :param model_save_dir: The directory to store the state at.
        :param state: The state containing the experiment state and the network. It's in the form of a dictionary
        object.
        """
        state['network'] = self.state_dict()
        #state['optimizer'] = self.optimizer.state_dict()
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
        #self.optimizer.load_state_dict(state['optimizer'])
        self.load_state_dict(state_dict=state_dict_loaded)
        return state