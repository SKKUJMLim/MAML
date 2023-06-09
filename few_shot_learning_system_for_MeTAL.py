import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from meta_neural_network_architectures import VGGReLUNormNetwork, ResNet12, MetaLossNetwork, LossAdapter
from inner_loop_optimizers import LSLRGradientDescentLearningRule


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

        # error를 잡기 위해 추가(JM)
        self.meta_loss = None
        self.meta_query_loss=None

        self.rng = set_torch_seed(seed=args.seed)

        if self.args.backbone == 'ResNet12':
            self.classifier = ResNet12(im_shape=self.im_shape, num_output_classes=self.args.
                                       num_classes_per_set,
                                       args=args, device=device, meta_classifier=True).to(device=self.device)
        else: # Conv-4
            self.classifier = VGGReLUNormNetwork(im_shape=self.im_shape, num_output_classes=self.args.
                                                 num_classes_per_set,
                                                 args=args, device=device, meta_classifier=True).to(device=self.device)

        self.task_learning_rate = args.init_inner_loop_learning_rate

        # Inner loop에서 최적화할 parameter를 설정
        # TODO: names_alpha_dict과 names_beta_dict에 대해서 감이 잘 오지 않는다. 오로지 ALFA를 위한 변수일까?
        self.inner_loop_optimizer = LSLRGradientDescentLearningRule(device=device,
                                                                    init_learning_rate=self.task_learning_rate,
                                                                    init_weight_decay=args.init_inner_loop_weight_decay,
                                                                    total_num_inner_loop_steps=self.args.number_of_training_steps_per_iter,
                                                                    use_learnable_weight_decay=self.args.alfa,
                                                                    use_learnable_learning_rates=(self.args.learnable_per_layer_per_step_inner_loop_learning_rate or self.args.alfa),
                                                                    alfa=self.args.alfa,
                                                                    random_init=self.args.random_init)

        # inner loop optimization process에 사용할 파라미터를 딕셔너리로 만든다
        ## outer loop optimization과 구별하기 위해서
        ## 코드가 중복된다
        names_weights_copy = self.get_inner_loop_parameter_dict(self.classifier.named_parameters())

        # ALFA와 비교했을 때, meta_loss가 생겼다
        if self.args.meta_loss:

            base_learner_num_layers = len(names_weights_copy)

            support_meta_loss_num_dim = base_learner_num_layers + 2 * self.args.num_classes_per_set + 1
            support_adapter_num_dim = base_learner_num_layers + 1
            query_num_dim = base_learner_num_layers + 1 + self.args.num_classes_per_set

            # TODO: Task status
            ## 1) support set loss의 평균(i-th itearation) 2) base learner의 weight의 layer-wise 평균 3) base-learner output value의 example wise 평균
            ## base learner f가 L-layer이고 output이 N차원이라면
            ## task state의 차원은 L+N+1
            ## 근데 왜 Meta_loss와 query_loss가 따로 있을까? -> Semi-supervised setting을 위해서

            self.meta_loss = MetaLossNetwork(support_meta_loss_num_dim, args=args, device=device).to(device=self.device)
            self.meta_query_loss = MetaLossNetwork(query_num_dim, args=args, device=device).to(device=self.device)

            self.meta_loss_adapter = LossAdapter(support_adapter_num_dim, num_loss_net_layers=2, args=args,
                                                 device=device).to(device=self.device)
            self.meta_query_loss_adapter = LossAdapter(query_num_dim, num_loss_net_layers=2, args=args,
                                                       device=device).to(device=self.device)

        # 각 paramter에 해당하는 alpha(learning rate), beta 초기값을 세팅해 loop만큼 놓는다
        self.inner_loop_optimizer.initialise(
            names_weights_dict=names_weights_copy)

        # Inner loop에서 최적화할 파라미터를 확인한다
        print("Inner Loop parameters")
        for key, value in self.inner_loop_optimizer.named_parameters():
            print(key, value.shape)
            ## 근데 alpha, beta 뿐이네? 가 아니라
            ## names_alpha_dict와 names_beta_dict에 각각 self.classifier의 파라미터를 배치했다.(norm_layer를 제외하고..)
            ## alpha에는 learning rate가 들어가고, beta에는 weight decay * learning rate로 초기화 되어있다

            # TODO: 왜? Conv Layer는 없을까?
            ## Classifier(CNN)의 파라미터도 같이 있어야하는거 아닌가? Outer-Loop parameters 처럼?
            ## MAMLFewShotClassifier에서 names_weights_copy를 가지고 있기 때문이다.
            ## 그저 print로 출력이 안되는 것일 뿐이다
            ## inner-loops는 net_forward로 loss, pred를 구한 후(순전파)
            ## apply_inner_loop_update에서 가중치가 업데이트 된다.

        self.use_cuda = args.use_cuda
        self.device = device
        self.args = args
        self.to(device)

        # outer loop를 확인한다
        print("Outer Loop parameters")
        for name, param in self.named_parameters():

            if param.requires_grad:
                print(name, param.shape, param.device, param.requires_grad)

        # TODO: Inner Loop와 Outer Loop를 확인해보니,
        ## Inner Loop의 parameter는 LSLRGradientDescentLearningRule에서 정의를 하고
        ## outer Loop의 Paramter는 여기 MAMLFewShotClassifier에서 정의를 했다
        ## 마치, Inner loop에서는 self.classifier에 파라미터를 update하지 않는 것처럼 보인다
        ## 그러나 MAML에서는 분명히 하지 않나?
        ## 본 코드의 backbone격인 MAML++도 names_learning_rates_dict이라는 변수에 learning rate를 담았다. 지켜볼 필요가 있다

        # ALFA
        if self.args.alfa:
            # input의 차원이 names_weights_copy 길이 x 2?
            ## 이 것은 무엇을 하려고 하는 것일까? -> alpha와 beta를 위한 것이다
            ## ALFA에서는 Inner loop interation동안 주어진 task에 적응할 수 있게 하는 Hyper Parmeter(learning rate, weight decay)를 생성한다
            num_layers = len(names_weights_copy)
            input_dim = num_layers * 2
            self.update_rule_learner = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(inplace=True),
                nn.Linear(input_dim, input_dim)
            ).to(device=self.device)

        learnable_params = list()

        self.optimizer = optim.Adam(self.trainable_parameters(), lr=args.meta_learning_rate, amsgrad=False)

        # lr_scheduler.CosineAnnealingLR => Cosine 그래프를 그리면서 learning rate가 진동하는 방식.
        ## learning rate가 단순히 감소하기 보다는 진동하면서 최적점을 찾아감
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

            self.device = torch.cuda.current_device() ##

        ## ===== __init__() end =====


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

    def apply_inner_loop_update(self, loss, names_weights_copy, generated_alpha_params, generated_beta_params,
                                use_second_order, current_step_idx, grads=None):
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

        # 보통 딥러닝에서는 미니배치+루프 조합을 사용해서 parameter들을 업데이트하는데, 한 루프에서 업데이트를 위해 loss.backward()를 호출하면 각 파라미터들의 .grad 값에 변화도가 저장된다
        # 이후 다음 루프에서 zero_grad()를 하지않고 역전파를 시키면 이전 루프에서 .grad에 저장된 값이 다음 루프의 업데이트에도 간섭을 해서 원하는 방향으로 학습이 되지 않음
        # 따라서 루프가 한번 돌고나서 역전파를 하기전에 반드시 zero_grad()로 .grad 값들을 0으로 초기화시킨 후 학습을 진행
        if num_gpus > 1:
            self.classifier.module.zero_grad(params=names_weights_copy)
        else:
            self.classifier.zero_grad(params=names_weights_copy)

        if grads is None:
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
                                                                     generated_alpha_params=generated_alpha_params,
                                                                     generated_beta_params=generated_beta_params,
                                                                     num_step=current_step_idx)

        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

        # TODO: 변한게 없는데..이게 뭐하는걸까?
        # print("apply_inner_loop_update before names_weights_copy == ", names_weights_copy.items())
        names_weights_copy = {
            name.replace('module.', ''): value.unsqueeze(0).repeat(
                [num_devices] + [1 for i in range(len(value.shape))]) for
            name, value in names_weights_copy.items()}
        # print("apply_inner_loop_update after names_weights_copy == ", names_weights_copy.items())

        return names_weights_copy

    def get_across_task_loss_metrics(self, total_losses, total_accuracies):
        losses = dict()

        losses['loss'] = torch.stack(total_losses)
        losses['accuracy'] = total_accuracies

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

        # print("x_support_set == " , len(x_support_set))
        # print("x_target_set == ", len(x_target_set))
        # print("y_support_set == ", len(y_support_set))
        # print("y_target_set == ", len(y_target_set))

        [b, ncs, spc] = y_support_set.shape

        self.num_classes_per_set = ncs

        total_losses = []
        total_accuracies = []
        total_support_accuracies = [[] for i in range(num_steps)]
        total_target_accuracies = [[] for i in range(num_steps)]

        # 이것은 무엇일까?
        per_task_target_preds = [[] for i in range(len(x_target_set))]

        # TODO: task를 구별할 수 있다는 task_id가 있다는 것은 task 별로 학습을 진행한다는 것 아닌가?
        for task_id, (x_support_set_task, y_support_set_task, x_target_set_task, y_target_set_task) in \
                enumerate(zip(x_support_set,
                              y_support_set,
                              x_target_set,
                              y_target_set)):

            # print("y_support_set_task len == ", len(y_support_set_task))
            # print("y_target_set_task len == ", len(y_target_set_task))

            task_losses = []
            task_accuracies = []
            per_step_support_accuracy = []
            per_step_target_accuracy = []

            # 이건 또 뭐지?
            per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector()

            # inner loop optimization process에 사용할 파라미터를 딕셔너리로 만든다
            names_weights_copy = self.get_inner_loop_parameter_dict(self.classifier.named_parameters())

            num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

            # TODO: 변한게 없는데..이게 뭐하는걸까?
            ## 아래 코드가 종종 보인다.. 뭘까?
            names_weights_copy = {
                name.replace('module.', ''): value.unsqueeze(0).repeat(
                    [num_devices] + [1 for i in range(len(value.shape))]) for
                name, value in names_weights_copy.items()}

            # MAML 코드를 구현하기 위해 추가 (JM)
            if self.args.meta_loss:
                names_loss_weights_copy_list = []
                names_query_loss_weights_copy_list = []
                names_loss_weights_copy = self.get_inner_loop_parameter_dict(self.meta_loss.named_parameters())
                names_query_loss_weights_copy = self.get_inner_loop_parameter_dict(self.meta_query_loss.named_parameters())

                names_loss_weights_copy = {
                    name.replace('module.', ''): value.unsqueeze(0).repeat(
                        [num_devices] + [1 for i in range(len(value.shape))]) for
                    name, value in names_loss_weights_copy.items()}

                names_query_loss_weights_copy = {
                    name.replace('module.', ''): value.unsqueeze(0).repeat(
                        [num_devices] + [1 for i in range(len(value.shape))]) for
                    name, value in names_query_loss_weights_copy.items()}
            else:
                names_loss_weights_copy = None
                names_query_loss_weights_copy = None

            n, s, c, h, w = x_target_set_task.shape

            x_support_set_task = x_support_set_task.view(-1, c, h, w)
            y_support_set_task = y_support_set_task.view(-1)
            x_target_set_task = x_target_set_task.view(-1, c, h, w)
            y_target_set_task = y_target_set_task.view(-1)

            # TODO: 왜 25, 75가 반복될까?
            #  당연하다. 5-way, 5-shot이면 support set의 개수는 25가 되고 query set의 개수는 75가 되는게 맞다.
            # print("x_support_set_task len == ", len(x_support_set_task))
            # print("y_support_set_task len == ", len(y_support_set_task))
            # print("x_target_set_task len == ", len(x_target_set_task))
            # print("y_target_set_task len == ", len(y_target_set_task))

            # MAML Outer-loop
            ## num_steps=args.number_of_training_steps_per_iter
            for num_step in range(num_steps):

                # MAML Inner-loop
                ## self.net_forward의 return 값 순서 : loss, preds, support_loss
                ## self.meta_loss가 None인 경우 (names_loss_weights_copy == None)이면 loss와 support_loss가 같다
                ## num_step은 inner-loop의 index를 뜻한다
                meta_loss, support_preds, support_loss = self.net_forward(x=x_support_set_task,
                                                                          y=y_support_set_task,
                                                                          weights=names_weights_copy,
                                                                          backup_running_statistics=
                                                                          True if (num_step == 0) else False,
                                                                          training=True, num_step=num_step,
                                                                          x_t=x_target_set_task,
                                                                          y_t=y_target_set_task, # 이것을 추가했는데, 넣어도 되는건가?
                                                                          meta_loss_weights=names_loss_weights_copy,
                                                                          meta_query_loss_weights=names_query_loss_weights_copy)

                generated_alpha_params = {}
                generated_beta_params = {}

                loss_grads = None

                if self.args.alfa:

                    loss_grads = torch.autograd.grad(meta_loss, names_weights_copy.values(),
                                                     create_graph=use_second_order)
                    per_step_task_embedding = []
                    for k, v in names_weights_copy.items():
                        per_step_task_embedding.append(v.mean())

                    for i in range(len(loss_grads)):
                        per_step_task_embedding.append(loss_grads[i].mean())

                    per_step_task_embedding = torch.stack(per_step_task_embedding)

                    # ALFA도 분석할 필요가 있다
                    generated_params = self.update_rule_learner(per_step_task_embedding)
                    num_layers = len(names_weights_copy)

                    generated_alpha, generated_beta = torch.split(generated_params, split_size_or_sections=num_layers)
                    g = 0
                    for key in names_weights_copy.keys():
                        generated_alpha_params[key] = generated_alpha[g]
                        generated_beta_params[key] = generated_beta[g]
                        g += 1

                # Inner loop의 loss 값과 pred 값으로 apply_inner_loop_update로 parameter를 update한다
                names_weights_copy = self.apply_inner_loop_update(loss=meta_loss,
                                                                  names_weights_copy=names_weights_copy,
                                                                  generated_beta_params=generated_beta_params,
                                                                  generated_alpha_params=generated_alpha_params,
                                                                  use_second_order=use_second_order,
                                                                  current_step_idx=num_step,
                                                                  grads=loss_grads)

                ## MAML++에서는 use_multi_step_loss_optimization이 true다
                if use_multi_step_loss_optimization and training_phase and epoch < self.args.multi_step_loss_num_epochs:
                    target_loss, target_preds, _ = self.net_forward(x=x_support_set_task,
                                                                    y=y_support_set_task, weights=names_weights_copy,
                                                                    backup_running_statistics=False, training=True,
                                                                    num_step=num_step,
                                                                    x_t=x_target_set_task,
                                                                    y_t=y_target_set_task)

                    task_losses.append(per_step_loss_importance_vectors[num_step] * target_loss)

                # TODO: Inner-loop안에서 query data에 대한 loss 함수의 함을 구한다
                ## 기억하자.
                ## MAML에서는 gradient decent를 할때, inner-loop에서 수행했던 query data에 대한 loss 값들의 합을 이용하여
                ## outer-loop에서 meta-learner의 weight를 update한다.
                else:
                    if num_step == (self.args.number_of_training_steps_per_iter - 1):
                        #  apply_inner_loop_update로 parameter를 update를 하고,
                        ## MAML++ 코드는 x_target_set_task로 net_forward 수행한다
                        target_loss, target_preds, _ = self.net_forward(x=x_support_set_task,
                                                                        y=y_support_set_task,
                                                                        weights=names_weights_copy,
                                                                        backup_running_statistics=False, training=True,
                                                                        num_step=num_step,
                                                                        x_t=x_target_set_task,
                                                                        y_t=y_target_set_task)

                        # 변수 명이 task_losses인것을 보면, task 당 loss를 구할 수 있다는 말일 것이다.
                        task_losses.append(target_loss)

                ### MAML inner-loop End

            per_task_target_preds[task_id] = target_preds.detach().cpu().numpy()
            _, predicted = torch.max(target_preds.data, 1)

            accuracy = predicted.float().eq(y_target_set_task.data.float()).cpu().float()

            # Query data에 대한 loss의 합을 구하는 부분
            task_losses = torch.sum(torch.stack(task_losses))
            total_losses.append(task_losses)
            total_accuracies.extend(accuracy)

            if not training_phase:
                if torch.cuda.device_count() > 1:
                    self.classifier.module.restore_backup_stats()
                else:
                    self.classifier.restore_backup_stats()

        losses = self.get_across_task_loss_metrics(total_losses=total_losses,
                                                   total_accuracies=total_accuracies)

        for idx, item in enumerate(per_step_loss_importance_vectors):
            losses['loss_importance_vector_{}'.format(idx)] = item.detach().cpu().numpy()

        return losses, per_task_target_preds

    def net_forward(self, x, y, weights, backup_running_statistics, training, num_step, meta_loss_weights=None,
                    x_t=None, y_t=None, meta_query_loss_weights=None):
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

        # TODO: torch.cat((x, x_t)??????
        ## x_t가 target이구나
        tmp_preds = self.classifier.forward(x=torch.cat((x, x_t), 0), params=weights,
                                            training=training,
                                            backup_running_statistics=backup_running_statistics, num_step=num_step)
        # print("tmp_preds == ", tmp_preds)
        support_preds = tmp_preds[:-x_t.size(0)]
        query_preds = tmp_preds[-x_t.size(0):]

        # print("support_preds == ", support_preds)
        # print("support_preds len== ", type(support_preds))

        # meta_loss == False인 경우
        ## support_loss와 loss가 같다
        if meta_loss_weights is None:
            loss = F.cross_entropy(input=tmp_preds, target=torch.cat((y, y_t), 0))
            preds = query_preds
            support_loss = loss

        else:
            support_task_state = []

            support_loss = F.cross_entropy(input=support_preds, target=y)
            support_task_state.append(support_loss)

            for v in weights.values():
                support_task_state.append(v.mean())

            support_task_state = torch.stack(support_task_state)
            adapt_support_task_state = (support_task_state - support_task_state.mean()) / (
                        support_task_state.std() + 1e-12)

            updated_meta_loss_weights = self.meta_loss_adapter(adapt_support_task_state, num_step, meta_loss_weights)

            support_y = torch.zeros(support_preds.shape).to(support_preds.device)
            support_y[torch.arange(support_y.size(0)), y] = 1
            support_task_state = torch.cat((
                support_task_state.view(1, -1).expand(support_preds.size(0), -1),
                support_preds,
                support_y
            ), -1)

            support_task_state = (support_task_state - support_task_state.mean()) / (support_task_state.std() + 1e-12)
            meta_support_loss = self.meta_loss(support_task_state, num_step,
                                               params=updated_meta_loss_weights).mean().squeeze()

            query_task_state = []
            for v in weights.values():
                query_task_state.append(v.mean())
            out_prob = F.log_softmax(query_preds)
            instance_entropy = torch.sum(torch.exp(out_prob) * out_prob, dim=-1)
            query_task_state = torch.stack(query_task_state)
            query_task_state = torch.cat((
                query_task_state.view(1, -1).expand(instance_entropy.size(0), -1),
                query_preds,
                instance_entropy.view(-1, 1)
            ), -1)

            query_task_state = (query_task_state - query_task_state.mean()) / (query_task_state.std() + 1e-12)
            updated_meta_query_loss_weights = self.meta_query_loss_adapter(query_task_state.mean(0), num_step,
                                                                           meta_query_loss_weights)

            meta_query_loss = self.meta_query_loss(query_task_state, num_step,
                                                   params=updated_meta_query_loss_weights).mean().squeeze()

            loss = support_loss + meta_query_loss + meta_support_loss

            preds = support_preds

        return loss, preds, support_loss

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

        # outer loop에서 실질적으로 inner loop를 호출하는 부분
        ## num_steps의 정체
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

    def meta_update(self, loss, task_idx):
        """
        Applies an outer loop update on the meta-parameters of the model.
        :param loss: The current crossentropy loss.
        """
        loss.backward()

        if task_idx == self.args.batch_size - 1:
            self.optimizer.step()

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

        stacked_loss = None
        stacked_acc = None
        self.optimizer.zero_grad()

        # Outer-loop
        for nt in range(self.args.batch_size):
            ## Meta-Batch
            x_support_set_t = x_support_set[nt:nt + 1]
            y_support_set_t = y_support_set[nt:nt + 1]
            x_target_set_t = x_target_set[nt:nt + 1]
            y_target_set_t = y_target_set[nt:nt + 1]

            data_batch = (x_support_set_t, x_target_set_t, y_support_set_t, y_target_set_t)


            ## outer-loop를 update하기 위해 loss 값을 구한다
            losses, per_task_target_preds = self.train_forward_prop(data_batch=data_batch, epoch=epoch)

            # 여기다가 curriculum을 추가하면 되려나? 그렇다면 inner loop는?>
            self.meta_update(loss=losses['loss'] / self.args.batch_size, task_idx=nt)

            if stacked_loss is None:
                stacked_loss = losses['loss'].detach()
                stacked_acc = losses['accuracy']
            else:
                stacked_loss = torch.cat((stacked_loss, losses['loss'].detach()), 0)
                stacked_acc = np.concatenate((stacked_acc, losses['accuracy']), 0)

        losses['loss'] = torch.mean(stacked_loss).item()
        losses['accuracy'] = np.mean(stacked_acc)

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
        losses['loss'] = torch.mean(losses['loss']).item()
        losses['accuracy'] = np.mean(losses['accuracy'])

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

