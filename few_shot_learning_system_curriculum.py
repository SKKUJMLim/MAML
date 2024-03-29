import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from meta_neural_network_architectures import VGGReLUNormNetwork, ResNet12
from inner_loop_optimizers import LSLRGradientDescentLearningRule

from utils.storage import save_statistics



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

        self.experiment_name = self.args.experiment_name
        self.comprehensive_loss_excel_create = True

        if self.args.backbone == 'ResNet12':
            self.classifier = ResNet12(im_shape=self.im_shape, num_output_classes=self.args.
                                       num_classes_per_set,
                                       args=args, device=device, meta_classifier=True).to(device=self.device)
        else:  # Conv-4
            self.classifier = VGGReLUNormNetwork(im_shape=self.im_shape, num_output_classes=self.args.
                                                 num_classes_per_set,
                                                 args=args, device=device, meta_classifier=True).to(device=self.device)

        self.task_learning_rate = args.init_inner_loop_learning_rate

        self.inner_loop_optimizer = LSLRGradientDescentLearningRule(device=device,
                                                                    init_learning_rate=self.task_learning_rate,
                                                                    init_weight_decay=args.init_inner_loop_weight_decay,
                                                                    total_num_inner_loop_steps=self.args.number_of_training_steps_per_iter,
                                                                    use_learnable_weight_decay=self.args.alfa,
                                                                    use_learnable_learning_rates=self.args.alfa,
                                                                    alfa=self.args.alfa,
                                                                    random_init=self.args.random_init)

        names_weights_copy = self.get_inner_loop_parameter_dict(self.classifier.named_parameters())
        #print("names_weights_copy == ", names_weights_copy)

        self.inner_loop_optimizer.initialise(names_weights_dict=names_weights_copy)

        if self.args.curriculum:

            ## input : loss, dropout loss, gradient, weight
            num_layers = len(names_weights_copy)
            input_dim = 2 + (num_layers * 2)

            self.curriculum_arbiter = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(inplace=True),
                nn.Linear(input_dim, 1)
            ).to(device=self.device)


        print("Inner Loop parameters")
        for key, value in self.inner_loop_optimizer.named_parameters():
            print(key, value.shape)
        print("=====================")

        self.use_cuda = args.use_cuda
        self.device = device
        self.args = args
        self.to(device)

        print("Outer Loop parameters")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.shape, param.device, param.requires_grad)
        print("=====================")


        self.optimizer = optim.Adam(self.trainable_parameters(), lr=args.meta_learning_rate, amsgrad=False)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.args.total_epochs,
                                                              eta_min=self.args.min_learning_rate)

        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            print(torch.cuda.device_count())
            if torch.cuda.device_count() > 1:
                self.to(torch.cuda.current_device())
                self.classifier = nn.DataParallel(module=self.classifier)
            else:
                self.to(torch.cuda.current_device())

            self.device = torch.cuda.current_device()  ##

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

    def apply_inner_loop_update(self, loss, names_weights_copy, use_second_order, current_step_idx):
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
                                                                     num_step=current_step_idx)

        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        names_weights_copy = {
            name.replace('module.', ''): value.unsqueeze(0).repeat(
                [num_devices] + [1 for i in range(len(value.shape))]) for
            name, value in names_weights_copy.items()}

        return names_weights_copy

    def get_across_task_loss_metrics(self, total_losses, total_accuracies):
        losses = dict()

        #print("total_losses == ", total_losses)

        losses['loss'] = torch.mean(torch.stack(total_losses))
        losses['accuracy'] = np.mean(total_accuracies)

        return losses

    def layer_wise_similarity(self, grad1, grad2):

        # grad1과 grad2의 차원이 같을 때만 사용 가능하다

        layerwise_sim_grads = []

        for i in range(len(grad1)):
            cos_sim = F.cosine_similarity(grad1[i], grad2[i])
            layerwise_sim_grads.append(cos_sim.mean())

        layerwise_sim_grads = torch.stack(layerwise_sim_grads)

        return layerwise_sim_grads

    def get_task_embeddings(self, x_support_set_task, y_support_set_task, names_weights_copy):
        # 내가 input으로 활용할 변수
        ## 1) support set을 통해 구한 loss
        ## 2) support set을 통해 구한 gradient (layer-wise mean)
        ## 3) dropout loss
        ## 4) weight norm (meta-learner) 이건 왜?? meta-learner의 norm 값이 필요한가..? meta-laerner와 base-learner의 weight를 생각해보자
        ### 즉, inner-loop 안에서 get_task_embeddings를 구현해야할거 같다.. MeTAL 처럼..쉬운일이다..

        per_step_task = []

        support_loss, support_preds, loss_with_dropout = self.net_forward(x=x_support_set_task,
                                                       y=y_support_set_task,
                                                       weights=names_weights_copy,
                                                       backup_running_statistics=True,
                                                       training=True, num_step=0)

        self.classifier.zero_grad(names_weights_copy)

        per_step_task.append(support_loss)
        per_step_task.append(loss_with_dropout)

        support_grads = torch.autograd.grad(support_loss, names_weights_copy.values(), create_graph=True)

        # Layer 별 Gradient 평균을 구한다
        for i in range(len(support_grads)):
            per_step_task.append(support_grads[i].mean())

        # Layer 별 Weight 평균을 구한다
        for k, v in names_weights_copy.items():
            per_step_task.append(v.mean())

        # 평균을 구하면 안될거 같다..
        ## Task-Specific Learner들 사이에 Conv1 ~ Conv4 Layer의 유사도가 높으니 마지막 FC Layer만 구해보자.
        # for k, v in names_weights_copy.items():
        #     if k == 'layer_dict.linear.weights':
        #         l1_norm = torch.norm(v, p=2, dim=1)

        per_step_task = torch.stack(per_step_task)

        return per_step_task

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

        total_losses = []
        total_accuracies = []
        per_task_target_preds = [[] for i in range(len(x_target_set))]
        self.classifier.zero_grad()
        task_accuracies = []

        # Outer-loop Start
        ## batch size만큼, 1 iteration을 수행한다.
        for task_id, (x_support_set_task, y_support_set_task, x_target_set_task, y_target_set_task) in enumerate(
                zip(x_support_set,
                    y_support_set,
                    x_target_set,
                    y_target_set)):

            # print("task_id == ", task_id)
            ## task_id ==  0
            ## task_id ==  1 이 반복된다
            ## batch_size가 2이기 때문이다. 즉 batch_size는 한번에 학습할 task의 수를 뜻한다

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

            comprehensive_losses = {}
            comprehensive_losses["epoch"] = epoch
            comprehensive_losses["task_id"] = task_id

            if training_phase:
                comprehensive_losses["phase"] = "train"
            else:
                comprehensive_losses["phase"] = "val"

            comprehensive_losses["num_steps"] = self.args.number_of_training_steps_per_iter

            for num_step in range(self.args.number_of_training_steps_per_iter):
                comprehensive_losses["support_loss_" + str(num_step)] = "null"
                comprehensive_losses["support_accuracy_" + str(num_step)] = "null"

            ## Inner-loop Start
            for num_step in range(num_steps):
                support_loss, support_preds, _= self.net_forward(
                    x=x_support_set_task,
                    y=y_support_set_task,
                    weights=names_weights_copy,
                    backup_running_statistics=
                    True if (num_step == 0) else False,
                    training=True, num_step=num_step)


                # print("support_loss == " , support_loss)
                comprehensive_losses["support_loss_" + str(num_step)] = support_loss.item()

                _, support_predicted = torch.max(support_preds.data, 1)

                support_accuracy = support_predicted.float().eq(y_support_set_task.data.float()).cpu().float()
                comprehensive_losses["support_accuracy_" + str(num_step)] = np.mean(list(support_accuracy))

                # task specific knowledge를 얻는 부분
                names_weights_copy = self.apply_inner_loop_update(loss=support_loss,
                                                                  names_weights_copy=names_weights_copy,
                                                                  use_second_order=use_second_order,
                                                                  current_step_idx=num_step)

                curriculum_loss = 0.0

                if use_multi_step_loss_optimization and training_phase and epoch < self.args.multi_step_loss_num_epochs:
                    target_loss, target_preds, _= self.net_forward(x=x_target_set_task,
                                                                 y=y_target_set_task, weights=names_weights_copy,
                                                                 backup_running_statistics=False, training=True,
                                                                 num_step=num_step)

                    task_losses.append(per_step_loss_importance_vectors[num_step] * target_loss)

                elif num_step == (self.args.number_of_training_steps_per_iter - 1):

                    # Inner-loop 결과를 바탕으로 Curriculum을 구성한다.
                    if self.args.curriculum:
                        per_step_task = self.get_task_embeddings(
                            x_support_set_task=x_support_set_task,
                            y_support_set_task=y_support_set_task,
                            names_weights_copy=names_weights_copy)
                        # apply_inner_loop_updater가 안됐을 때 weight 값을 넣고있다.
                        # 반드시 수정해야할 부분이다.
                        # 그러나 지금은 curriculum_loss 값이 바뀌지 않는 문제를 해결하는게 더욱 시급하다

                        per_step_task = (per_step_task - per_step_task.mean()) / (per_step_task.std() + 1e-12)

                        curriculum_loss = self.curriculum_arbiter(per_step_task) + 1.0

                        # Excel에 기록하자
                        losses_List = per_step_task[:2]
                        comprehensive_losses["dropout_losses"] = losses_List[1].item()
                        gradient_List = per_step_task[2:12]

                        a=1
                        for grad in gradient_List:
                            comprehensive_losses["gradient_layer_" + str(a)] = grad.item()
                            a=a+1

                        a=1
                        weight_List = per_step_task[12:22]
                        for weight in weight_List:
                            comprehensive_losses["weight_layer_" + str(a)]  = weight.item()
                            a=a+1

                        comprehensive_losses["curriculum_loss" + str(num_step)] = curriculum_loss.item()
                        #### Excel 기록

                    target_loss, target_preds, _ = self.net_forward(x=x_target_set_task,
                                                                 y=y_target_set_task, weights=names_weights_copy,
                                                                 backup_running_statistics=False, training=True,
                                                                 num_step=num_step, isFocalLoss=True)



                    # curriculum loss보다 크면 loss를 0으로 만들어서 가중치 업데이트를 막는다.
                    if self.args.curriculum:
                        if target_loss > curriculum_loss:
                            target_loss = torch.zeros(1).to(device=self.device)
                        else:
                            pass
                            #comprehensive_losses["focal_loss" + str(num_step)] = focal_loss.item()
                            #target_loss = focal_loss

                    task_losses.append(target_loss)
                    comprehensive_losses["target_loss_" + str(num_step)] = target_loss.item()

                    _, target_predicted = torch.max(target_preds.data, 1)
                    target_accuracy = target_predicted.float().eq(y_target_set_task.data.float()).cpu().float()
                    comprehensive_losses["target_accuracy_" + str(num_step)] = np.mean(list(target_accuracy))
                ## Inner-loop END

            # Inner-loop 결과를 csv로 생성한다.
            if self.comprehensive_loss_excel_create:
                save_statistics(experiment_name=self.experiment_name,
                                line_to_add=list(comprehensive_losses.keys()),
                                filename=self.experiment_name+".csv", create=True)
                self.comprehensive_loss_excel_create = False
                save_statistics(experiment_name=self.experiment_name,
                                line_to_add=list(comprehensive_losses.values()),
                                filename=self.experiment_name+".csv", create=False)
            else:
                save_statistics(experiment_name=self.experiment_name,
                                line_to_add=list(comprehensive_losses.values()),
                                filename=self.experiment_name+".csv", create=False)


            per_task_target_preds[task_id] = target_preds.detach().cpu().numpy()
            _, predicted = torch.max(target_preds.data, 1)

            accuracy = predicted.float().eq(y_target_set_task.data.float()).cpu().float()

            # batch에 대한 학습이 끝나고, acc와 loss를 기록
            ## torch.sum을 더한다는 의미로 받아들여서는 안된다.. tensor로 바꿔준다는 것이다.
            task_losses = torch.sum(torch.stack(task_losses))
            total_losses.append(task_losses)
            total_accuracies.extend(accuracy)

            if not training_phase:
                self.classifier.restore_backup_stats()

            # Outer-loop End

        # 왜 평균을 내고 있을까?
        ## iteration (task 1, 2)에 대한 평균
        losses = self.get_across_task_loss_metrics(total_losses=total_losses,
                                                   total_accuracies=total_accuracies)

        for idx, item in enumerate(per_step_loss_importance_vectors):
            losses['loss_importance_vector_{}'.format(idx)] = item.detach().cpu().numpy()

        return losses, per_task_target_preds

    def net_forward(self, x, y, weights, backup_running_statistics, training, num_step, isFocalLoss=False):
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
        preds = self.classifier.forward(x=x, params=weights,
                                        training=training,
                                        backup_running_statistics=backup_running_statistics, num_step=num_step, isDropout=False)

        loss = F.cross_entropy(input=preds, target=y, reduction='none')
        focal_loss = 0.0

        if isFocalLoss:
            pt = torch.exp(-loss)
            alpha = 1
            gamma = 2
            focal_loss = (alpha * (1 - pt) ** gamma * loss).mean()  # mean over the batch


        loss = loss.mean()

        preds_with_Dropout = self.classifier.forward(x=x, params=weights,
                                                     training=training,
                                                     backup_running_statistics=backup_running_statistics,
                                                     num_step=num_step, isDropout=True)
        loss_with_dropout = F.cross_entropy(input=preds_with_Dropout, target=y)

        return loss, preds, loss_with_dropout

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
        losses, per_task_target_preds = self.forward(data_batch=data_batch, epoch=epoch,
                                                     use_second_order=self.args.second_order and
                                                                      epoch > self.args.first_order_to_second_order_epoch,
                                                     use_multi_step_loss_optimization=self.args.use_multi_step_loss_optimization,
                                                     num_steps=self.args.number_of_training_steps_per_iter,
                                                     training_phase=True)
        return losses, per_task_target_preds

    def evaluation_forward_prop(self, data_batch, epoch):
        """
        Runs an outer loop evaluation forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses, per_task_target_preds = self.forward(data_batch=data_batch, epoch=epoch, use_second_order=False,
                                                     use_multi_step_loss_optimization=True,
                                                     num_steps=self.args.number_of_evaluation_steps_per_iter,
                                                     training_phase=False)

        return losses, per_task_target_preds

    def meta_update(self, loss):
        """
        Applies an outer loop update on the meta-parameters of the model.
        :param loss: The current crossentropy loss.
        """

        # 가중치 업데이트 확인용 변수
        prev_weights = {}
        for name, param in self.curriculum_arbiter.named_parameters():
            prev_weights[name] = param.data.clone()

        self.optimizer.zero_grad()

        loss.backward()

        # if 'imagenet' in self.args.dataset_name:
        #     for name, param in self.classifier.named_parameters():
        #         if param.requires_grad:
        #             param.grad.data.clamp_(-10, 10)  # not sure if this is necessary, more experiments are needed

        self.optimizer.step()

        ## 가중치 업데이트 확인
        for name, param in self.curriculum_arbiter.named_parameters():
            if not torch.equal(prev_weights[name], param.data):
                print(f"{name} 가중치가 업데이트되었습니다.")
                prev_weights[name] = param.data.clone()

    def check_weight_update(self, initial_state, updated_state):
        # 가중치 변화 확인
        for name, param in updated_state.items():
            if not torch.equal(param, initial_state[name]):
                print(f"{name} 가중치가 업데이트되었습니다.")

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

        losses, per_task_target_preds = self.train_forward_prop(data_batch=data_batch, epoch=epoch)

        # 여기서 update를 생략하면 된다
        if not torch.eq(losses['loss'], torch.zeros(1).to(device=self.device)):
            ## loss에 current epoch \ total epoch을 지수로 취하자
            self.meta_update(loss=losses['loss'])

        losses['learning_rate'] = self.scheduler.get_lr()[0]
        self.optimizer.zero_grad()
        self.zero_grad()

        return losses, per_task_target_preds

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

        losses, per_task_target_preds = self.evaluation_forward_prop(data_batch=data_batch, epoch=self.current_epoch)

        if not torch.eq(losses['loss'], torch.zeros(1).to(device=self.device)):
            self.meta_update(loss=losses['loss'])

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
        self.load_state_dict(state_dict=state_dict_loaded)
        return state