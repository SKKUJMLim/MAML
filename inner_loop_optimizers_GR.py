import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.storage import save_statistics
import os


class GradientDescentLearningRule(nn.Module):
    """Simple (stochastic) gradient descent learning rule.
    For a scalar error function `E(p[0], p_[1] ... )` of some set of
    potentially multidimensional parameters this attempts to find a local
    minimum of the loss function by applying updates to each parameter of the
    form
        p[i] := p[i] - learning_rate * dE/dp[i]
    With `learning_rate` a positive scaling parameter.
    The error function used in successive applications of these updates may be
    a stochastic estimator of the true error function (e.g. when the error with
    respect to only a subset of data-points is calculated) in which case this
    will correspond to a stochastic gradient descent learning rule.
    """

    def __init__(self, device, args, learning_rate=1e-3):
        """Creates a new learning rule object.
        Args:
            learning_rate: A postive scalar to scale gradient updates to the
                parameters by. This needs to be carefully set - if too large
                the learning dynamic will be unstable and may diverge, while
                if set too small learning will proceed very slowly.
        """
        super(GradientDescentLearningRule, self).__init__()
        assert learning_rate > 0., 'learning_rate should be positive.'
        self.device = device


        self.learning_rate = learning_rate

        self.args = args
        self.norm_information = {}
        self.innerloop_excel = True

    def update_params(self, names_weights_dict, names_grads_wrt_params_dict, generated_alpha_params, num_step, current_iter, training_phase):
        """Applies a single gradient descent update to all parameters.
        All parameter updates are performed using in-place operations and so
        nothing is returned.
        Args:
            grads_wrt_params: A list of gradients of the scalar loss function
                with respect to each of the parameters passed to `initialise`
                previously, with this list expected to be in the same order.
        """

        updated_names_weights_dict = dict()

        #### 추가 ####
        if current_iter == 'test':
            try:
                self.norm_information['current_iter'] = os.environ["TEST_DATASET"]
            except:
                self.norm_information['current_iter'] = self.args.dataset_name
        else:
            self.norm_information['current_iter'] = current_iter
        ########

        if training_phase:
            self.norm_information["phase"] = "train"
        else:
            self.norm_information["phase"] = "val"

        self.norm_information['num_step'] = num_step

        for key in names_grads_wrt_params_dict.keys():

            #####
            if self.args.arbiter:

                self.norm_information[key + "_alpha"] = generated_alpha_params[key].item()

                applied_gradient = generated_alpha_params[key] *  \
                                   names_grads_wrt_params_dict[key] / torch.norm(names_grads_wrt_params_dict[key])

                self.norm_information[key + "_grad_mean"] = torch.mean(applied_gradient).item()
                self.norm_information[key + "_grad_L1norm"] = torch.norm(applied_gradient, p=1).item()
                self.norm_information[key + "_grad_L2norm"] = torch.norm(applied_gradient, p=2).item()
                self.norm_information[key + "_grad_var"] = torch.var(applied_gradient).item()

                updated_names_weights_dict[key] = names_weights_dict[key] - self.learning_rate * applied_gradient

                # updated_names_weights_dict[key] = names_weights_dict[key] - self.learning_rate * \
                #                                   generated_alpha_params[key] *  names_grads_wrt_params_dict[key] / torch.norm(names_grads_wrt_params_dict[key])
            else:

                # MAML
                self.norm_information[key + "_grad_mean"] = torch.mean(names_grads_wrt_params_dict[key]).item()
                self.norm_information[key + "_grad_L1norm"] = torch.norm(names_grads_wrt_params_dict[key], p=1).item()
                self.norm_information[key + "_grad_L2norm"] = torch.norm(names_grads_wrt_params_dict[key], p=2).item()
                self.norm_information[key + "_grad_var"] = torch.var(names_grads_wrt_params_dict[key]).item()

                updated_names_weights_dict[key] = names_weights_dict[key] - self.learning_rate * \
                                                  names_grads_wrt_params_dict[key]
            ############## if문 종료

            ## updated_names_weights_dict를 기록해야할까?
            self.norm_information[key + "_weight_mean"] = torch.mean(names_weights_dict[key]).item()
            self.norm_information[key + "_weight_L1norm"] = torch.norm(names_weights_dict[key], p=1).item()
            self.norm_information[key + "_weight_L2norm"] = torch.norm(names_weights_dict[key], p=2).item()
            self.norm_information[key + "_weight_var"] = torch.var(names_weights_dict[key]).item()

        if os.path.exists(self.args.experiment_name + '/' + self.args.experiment_name + "_inner_loop.csv"):
            self.innerloop_excel = False

        if self.innerloop_excel:
            save_statistics(experiment_name=self.args.experiment_name,
                            line_to_add=list(self.norm_information.keys()),
                            filename=self.args.experiment_name + "_inner_loop.csv", create=True)
            self.innerloop_excel = False
            save_statistics(experiment_name=self.args.experiment_name,
                            line_to_add=list(self.norm_information.values()),
                            filename=self.args.experiment_name + "_inner_loop.csv", create=False)
        else:
            save_statistics(experiment_name=self.args.experiment_name,
                            line_to_add=list(self.norm_information.values()),
                            filename=self.args.experiment_name + "_inner_loop.csv", create=False)

        return updated_names_weights_dict


class LSLRGradientDescentLearningRule(nn.Module):
    """Simple (stochastic) gradient descent learning rule.
    For a scalar error function `E(p[0], p_[1] ... )` of some set of
    potentially multidimensional parameters this attempts to find a local
    minimum of the loss function by applying updates to each parameter of the
    form
        p[i] := p[i] - learning_rate * dE/dp[i]
    With `learning_rate` a positive scaling parameter.
    The error function used in successive applications of these updates may be
    a stochastic estimator of the true error function (e.g. when the error with
    respect to only a subset of data-points is calculated) in which case this
    will correspond to a stochastic gradient descent learning rule.
    """

    def __init__(self, device, args, total_num_inner_loop_steps, use_learnable_learning_rates, init_learning_rate=1e-3):
        """Creates a new learning rule object.
        Args:
            init_learning_rate: A postive scalar to scale gradient updates to the
                parameters by. This needs to be carefully set - if too large
                the learning dynamic will be unstable and may diverge, while
                if set too small learning will proceed very slowly.
        """
        super(LSLRGradientDescentLearningRule, self).__init__()
        print(init_learning_rate)
        assert init_learning_rate > 0., 'learning_rate should be positive.'

        self.init_learning_rate = torch.ones(1) * init_learning_rate
        self.init_learning_rate.to(device)
        self.total_num_inner_loop_steps = total_num_inner_loop_steps
        self.use_learnable_learning_rates = use_learnable_learning_rates

        self.args = args

        self.norm_information = {}
        self.innerloop_excel = True

    def initialise(self, names_weights_dict):
        self.names_learning_rates_dict = nn.ParameterDict()
        for idx, (key, param) in enumerate(names_weights_dict.items()):
            self.names_learning_rates_dict[key.replace(".", "-")] = nn.Parameter(
                data=torch.ones(self.total_num_inner_loop_steps + 1) * self.init_learning_rate,
                requires_grad=self.use_learnable_learning_rates)

    def update_params(self, names_weights_dict, names_grads_wrt_params_dict, generated_alpha_params, num_step, current_iter, training_phase, tau=0.1):
        """Applies a single gradient descent update to all parameters.
        All parameter updates are performed using in-place operations and so
        nothing is returned.
        Args:
            grads_wrt_params: A list of gradients of the scalar loss function
                with respect to each of the parameters passed to `initialise`
                previously, with this list expected to be in the same order.
        """

        updated_names_weights_dict = dict()

        self.norm_information['current_iter'] = current_iter

        if training_phase:
            self.norm_information["phase"] = "train"
        else:
            self.norm_information["phase"] = "val"

        self.norm_information['num_step'] = num_step

        for key in names_grads_wrt_params_dict.keys():

            self.norm_information[key + "_grad_mean"] = torch.mean(names_grads_wrt_params_dict[key]).item()
            self.norm_information[key + "_grad_L1norm"] = torch.norm(names_grads_wrt_params_dict[key], p=1).item()
            self.norm_information[key + "_grad_L2norm"] = torch.norm(names_grads_wrt_params_dict[key], p=2).item()
            self.norm_information[key + "_weight_mean"] = torch.mean(names_weights_dict[key]).item()
            self.norm_information[key + "_weight_L1norm"] = torch.norm(names_weights_dict[key], p=1).item()
            self.norm_information[key + "_weight_L2norm"] = torch.norm(names_weights_dict[key], p=2).item()

            if self.args.arbiter:

                self.norm_information[key + "_alpha"] = generated_alpha_params[key].item()

                updated_names_weights_dict[key] = names_weights_dict[key] - \
                                                  self.names_learning_rates_dict[key.replace(".", "-")][num_step] * \
                                                  generated_alpha_params[key] * (
                                                              names_grads_wrt_params_dict[key] / torch.norm(
                                                          names_grads_wrt_params_dict[key]))

                ##코드짜는중
                # if 'linear' in key:
                #     updated_names_weights_dict[key] = names_weights_dict[key] - \
                #                                       self.names_learning_rates_dict[key.replace(".", "-")][num_step] * \
                #                                       (names_grads_wrt_params_dict[key] / torch.norm(names_grads_wrt_params_dict[key]))
                # else:
                #     updated_names_weights_dict[key] = names_weights_dict[key] - \
                #                                       self.names_learning_rates_dict[key.replace(".", "-")][num_step] * \
                #                                       generated_alpha_params[key] * (names_grads_wrt_params_dict[key] / torch.norm(names_grads_wrt_params_dict[key]))

            else:
                updated_names_weights_dict[key] = names_weights_dict[key] - \
                                                  self.names_learning_rates_dict[key.replace(".", "-")][num_step] * \
                                                  names_grads_wrt_params_dict[key]
                # if self.args.SWA:
                #     #if num_step % 2 == 0:
                #     alpha = 1.0 / (num_step + 1)
                #     updated_names_weights_dict[key] = updated_names_weights_dict[key] * (1.0 - alpha)
                #     updated_names_weights_dict[key] = updated_names_weights_dict[key] + (names_weights_dict[key] * alpha)

        if os.path.exists(self.args.experiment_name + '/' + self.args.experiment_name + "_inner_loop.csv"):
            self.innerloop_excel = False

        if self.innerloop_excel:
            save_statistics(experiment_name=self.args.experiment_name,
                            line_to_add=list(self.norm_information.keys()),
                            filename=self.args.experiment_name + "_inner_loop.csv", create=True)
            self.innerloop_excel = False
            save_statistics(experiment_name=self.args.experiment_name,
                            line_to_add=list(self.norm_information.values()),
                            filename=self.args.experiment_name + "_inner_loop.csv", create=False)
        else:
            save_statistics(experiment_name=self.args.experiment_name,
                            line_to_add=list(self.norm_information.values()),
                            filename=self.args.experiment_name + "_inner_loop.csv", create=False)

        return updated_names_weights_dict