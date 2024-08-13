from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, \
    AdaptiveAvgPool2d, Sequential, Module

from torch.autograd import Function

import numpy as np
import scipy as sp
import scipy.linalg as linalg

from meta_neural_network_architectures import extract_top_level_dict


def skewness(tensor):
    # 데이터의 크기
    n = tensor.size(0)

    # 평균 계산
    mean = torch.mean(tensor)

    # 표준편차 계산
    std_dev = torch.std(tensor, unbiased=True)

    # 3차 중심적률 계산
    skewness_numerator = torch.sum((tensor - mean) ** 3) / n

    # 비대칭도 계산
    skewness_value = skewness_numerator / (std_dev ** 3)

    return skewness_value


def pearsons_second_skewness_coefficient(tensor):
    # 평균 계산
    mean = torch.mean(tensor)

    # 중앙값 계산
    median = torch.median(tensor)

    # 표준편차 계산
    std_dev = torch.std(tensor, unbiased=True)

    # 피어슨의 두 번째 비대칭 계수 계산
    skewness_value = 3 * (mean - median) / std_dev

    return skewness_value

def knowledge_distillation_loss(student_logit, teacher_logit, labels, label_loss_weight=1, soft_label_loss_weight=1, Temperature=1.0):
    """
    Knowledge Distillation Loss를 계산하는 함수
    outputs_student: 작은 모델의 출력
    outputs_teacher: 큰 모델의 출력
    labels: 실제 라벨
    T: temperature (soft target의 'softness'를 조절하는 파라미터)
    alpha: 손실 함수의 가중치 조절 파라미터
    """

    # 손실 함수 정의
    criterion = nn.CrossEntropyLoss()

    # Knowledge Distillation 손실 계산
    soft_teacher_outputs = nn.functional.softmax(teacher_logit / Temperature, dim=1)
    soft_student_outputs = nn.functional.log_softmax(student_logit / Temperature, dim=1)
    distillation_loss = nn.KLDivLoss()(soft_student_outputs, soft_teacher_outputs) * (Temperature * Temperature)

    classification_loss = criterion(student_logit, labels)

    # 총 손실 계산 (1. - alpha)
    loss = label_loss_weight * classification_loss + soft_label_loss_weight * distillation_loss

    return loss


def KL_divergence(student_logit, teacher_logit, Temperature):

    soft_teacher_outputs = nn.functional.softmax(teacher_logit / Temperature, dim=1)
    soft_student_outputs = nn.functional.log_softmax(student_logit / Temperature, dim=1)
    distillation_loss = nn.KLDivLoss()(soft_student_outputs, soft_teacher_outputs) * (Temperature * Temperature)

    return distillation_loss

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy w/ smoothing or soft targets

    Hacked together by / Copyright 2021 Ross Wightman
    """

    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1, reduction=None):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        self.reduction = reduction

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        else:
            loss = loss.mean()

        return loss


class SoftmaxLoss(nn.Module):
    def __init__(self, num_classes, embedding_size):
        """
        Regular softmax loss (1 fc layer without bias + CrossEntropyLoss)
        Args:
            num_classes: The number of classes in your training dataset
            embedding_size: The size of the embeddings that you pass into
        """
        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size

        self.W = torch.nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.xavier_normal_(self.W)

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (None, embedding_size)
            labels: (None,)
        Returns:
            loss: scalar
        """
        logits = F.linear(embeddings, self.W)
        return nn.CrossEntropyLoss()(logits, labels)

class ArcFace(nn.Module):

    # 참고
    # https://www.kaggle.com/code/kuposatina/arcface-test-2
    # https://www.kaggle.com/code/debarshichanda/pytorch-arcface-gem-pooling-starter
    # https://www.kaggle.com/code/nanguyen/arcface-loss

    """Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            device_id: the ID of GPU where the model will be trained by model parallel.
                       if device_id=None, it will be trained on CPU without model parallel.
            s: norm of input feature
            m: margin
            cos(theta+m)
    """


    def __init__(self, in_features, out_features, args, easy_margin=False):
        # 원본 : s=64.0, m=0.50
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.m = args.margin
        self.s = args.rescale

        print("=====CurricularFace====")
        print("margin == ", self.m)
        print("rescale == ", self.s)

        # kernerl은 weight를 뜻한다
        self.kernel = Parameter(torch.FloatTensor(in_features, out_features))
        # nn.init.xavier_uniform_(self.kernel)
        #nn.init.normal_(self.kernel, std=0.01)
        nn.init.xavier_uniform_(self.kernel)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, embbedings, label, params=None):

        embbedings = l2_norm(embbedings, axis=1)

        if params is not None:
            # param이 지정될 경우 (inner-loop)
            param_dict = extract_top_level_dict(current_dict=params)
            kernel = param_dict['kernel']
        else:
            kernel = self.kernel

        kernel_norm = l2_norm(kernel, axis=0)

        # deep feature x weight
        cos_theta = torch.mm(embbedings, kernel_norm)
        # embbedings, kernel_norm 모두 l2 norm을 통해 scale을 1로 조정했기 때문에, 분모가 없는 것이다.
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability

        #print("cos_theta == ", cos_theta)
        #print("embbedings.size(0) == ", embbedings.size(0)) #25

        with torch.no_grad():
            origin_cos = cos_theta.clone()

        target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))

        # cos(A+B) = cosA*CosB - SinA*SinB
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)

        if self.easy_margin:
            final_target_logit = torch.where(target_logit > 0, cos_theta_m, target_logit)
        else:
            # torch.where(condition, x, y) → condition에 따라 x 또는 y에서 선택한 요소의 텐서를 반환
            # 조건이 왜 target_logit > self.th일까?
            final_target_logit = torch.where(target_logit > self.th, cos_theta_m, target_logit - self.mm)

        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)

        output = cos_theta * self.s

        return output, origin_cos * self.s


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


class CurricularFace(nn.Module):
    def __init__(self, in_features, out_features, args):
        super(CurricularFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = args.margin
        self.s = args.rescale

        print("=====CurricularFace====")
        print("margin == ", self.m)
        print("rescale == ", self.s)

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.threshold = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        self.kernel = Parameter(torch.Tensor(in_features, out_features))
        self.register_buffer('t', torch.zeros(1))
        #nn.init.normal_(self.kernel, std=0.01)
        nn.init.xavier_uniform_(self.kernel)

    def forward(self, embbedings, label, params=None):
        embbedings = l2_norm(embbedings, axis=1)

        if params is not None:
            # param이 지정될 경우 (inner-loop)
            param_dict = extract_top_level_dict(current_dict=params)
            kernel = param_dict['kernel']
        else:
            kernel = self.kernel

        kernel_norm = l2_norm(kernel, axis=0)

        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return output, origin_cos * self.s

class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, eps = 1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

class OLELoss(Function):
    @staticmethod
    def forward(ctx, X, y):

        X = X.cpu().numpy()
        y = y.cpu().numpy()

        classes = np.unique(y)
        C = classes.size

        N, D = X.shape

        lambda_ = 1.
        DELTA = 1.

        # gradients initialization
        Obj_c = 0
        dX_c = np.zeros((N, D))
        Obj_all = 0;
        dX_all = np.zeros((N, D))

        eigThd = 1e-6  # threshold small eigenvalues for a better subgradient

        # compute objective and gradient for first term \sum ||TX_c||*
        for c in classes:
            A = X[y == c, :]

            # SVD
            U, S, V = sp.linalg.svd(A, full_matrices=False, lapack_driver='gesvd')

            V = V.T
            nuclear = np.sum(S);

            ## L_c = max(DELTA, ||TY_c||_*)-DELTA

            if nuclear > DELTA:
                Obj_c += nuclear;

                # discard small singular values
                r = np.sum(S < eigThd)
                uprod = U[:, 0:U.shape[1] - r].dot(V[:, 0:V.shape[1] - r].T)

                dX_c[y == c, :] += uprod
            else:
                Obj_c += DELTA

        # compute objective and gradient for secon term ||TX||*

        U, S, V = sp.linalg.svd(X, full_matrices=False, lapack_driver='gesvd')  # all classes

        V = V.T

        Obj_all = np.sum(S);

        r = np.sum(S < eigThd)

        uprod = U[:, 0:U.shape[1] - r].dot(V[:, 0:V.shape[1] - r].T)

        dX_all = uprod

        obj = (Obj_c - lambda_ * Obj_all) / N * np.float(lambda_)

        dX = (dX_c - lambda_ * dX_all) / N * np.float(lambda_)

        dX = torch.FloatTensor(dX)

        ctx.save_for_backward(dX)

        return torch.FloatTensor([float(obj)]).cuda()

    @staticmethod
    def backward(ctx, grad_output):
        dX = ctx.saved_tensors[0]

        # print(dX)

        return dX.cuda(), None

# 특정 레이어의 그래디언트 norm 값을 크게 만들기 위한 커스텀 손실 함수 정의
class CustomLoss(nn.Module):
    def __init__(self, model, layer_name, weight=0.1):
        super(CustomLoss, self).__init__()
        self.layer_name = layer_name
        self.weight = weight
        self.model = model

    def forward(self, output, target):
        layer_output = self.model._modules[self.layer_name].out_features
        custom_loss = self.weight * torch.norm(layer_output, p=2)  # L2 노름을 사용하고, 가중치를 곱하여 반환합니다.
        return custom_loss