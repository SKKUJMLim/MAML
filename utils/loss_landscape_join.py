import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
import copy
from mypyhessian import my_hessian
import torch.nn.functional as F

import os


class landscape(nn.Module):
    def __init__(self, model, model2, args):
        super(landscape, self).__init__()

        self.model = model
        self.model2 = model2
        self.args = args

    def get_params(self, model_orig, model_perb, direction, alpha):
        i=0
        for m_orig, m_perb in zip(model_orig.parameters(), model_perb.parameters()):
            if m_orig.requires_grad:
                m_perb.data = m_orig.data + alpha * direction[i]
                i = i+1
        return model_perb

    def show_2djoin(self, inputs, targets, title):

        model = self.model.cuda()
        model2 = self.model2.cuda()

        print("inputs shape== " ,inputs.shape)

        inputs, targets = inputs.cuda(), targets.cuda()

        hessian_comp = my_hessian.my_hessian(model, data=(inputs, targets), cuda=True)
        hessian_comp2 = my_hessian.my_hessian(model2, data=(inputs, targets), cuda=True)

        top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()
        top_eigenvalues2, top_eigenvector2 = hessian_comp2.eigenvalues()

        lams = np.linspace(-0.5, 0.5, 21).astype(np.float32)

        model_perb = copy.deepcopy(model)
        model_perb.eval()
        model_perb.cuda()

        model2_perb = copy.deepcopy(model2)
        model2_perb.eval()
        model2_perb.cuda()

        loss_list = []
        loss_list2 = []

        for lam in lams:
            model_perb = self.get_params(model, model_perb, top_eigenvector[0], lam)
            preds, _ = model_perb.forward(x=inputs, num_step=5)
            loss = F.cross_entropy(input=preds, target=targets)
            loss_list.append(loss.item())

            model2_perb = self.get_params(model2, model2_perb, top_eigenvector2[0], lam)
            preds, _ = model2_perb.forward(x=inputs, num_step=5)
            loss = F.cross_entropy(input=preds, target=targets)
            loss_list2.append(loss.item())

        self.save_landscape_2dimage(lams, loss_list, loss_list2, title)

    def show_3djoin(self, inputs, targets, title):

        loss_list = []
        loss_list2 = []

        model = self.model.cuda()
        model2 = self.model2.cuda()

        inputs, targets = inputs.cuda(), targets.cuda()

        hessian_comp = my_hessian.my_hessian(model, data=(inputs, targets), cuda=True)
        hessian_comp2 = my_hessian.my_hessian(model2, data=(inputs, targets), cuda=True)

        # get the top1, top2 eigenvectors
        top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)
        top_eigenvalues2, top_eigenvector2 = hessian_comp2.eigenvalues(top_n=2)

        lams1 = np.linspace(-0.5, 0.5, 31).astype(np.float32)
        lams2 = np.linspace(-0.5, 0.5, 31).astype(np.float32)

        model_perb1 = copy.deepcopy(model)
        model_perb1.eval()
        model_perb1.cuda()

        model_perb2 = copy.deepcopy(model)
        model_perb2.eval()
        model_perb2.cuda()

        model2_perb1 = copy.deepcopy(model2)
        model2_perb1.eval()
        model2_perb1.cuda()

        model2_perb2 = copy.deepcopy(model2)
        model2_perb2.eval()
        model2_perb2.cuda()

        for lam1 in lams1:
            for lam2 in lams2:
                model_perb1 = self.get_params(model, model_perb1, top_eigenvector[0], lam1)
                model_perb2 = self.get_params(model_perb1, model_perb2, top_eigenvector[1], lam2)

                preds = model_perb2.forward(x=inputs, num_step=5)
                loss = F.cross_entropy(input=preds, target=targets)
                loss_list.append((lam1, lam2, loss.item()))


                model2_perb1 = self.get_params(model2, model2_perb1, top_eigenvector2[0], lam1)
                model2_perb2 = self.get_params(model2_perb1, model2_perb2, top_eigenvector2[1], lam2)

                preds = model2_perb2.forward(x=inputs, num_step=5)
                loss = F.cross_entropy(input=preds, target=targets)
                loss_list2.append((lam1, lam2, loss.item()))

        loss_list = np.array(loss_list)
        loss_list2 = np.array(loss_list2)

        self.save_landscape_3dimage(loss_list, loss_list2, title)

    def save_landscape_2dimage(self, lams, loss_list, loss_list2, title):

        fig, ax = plt.subplots()
        # plt.plot(lams, loss_list, lams, loss_list2, 'r-', lw=3)
        plt.plot(lams, loss_list, 'b-', label='MAML', lw=1)
        plt.plot(lams, loss_list2, 'r-', label='Meta-GNA', lw=1)
        plt.ylabel('Loss', fontsize=14)
        plt.xlabel('Perturbation', fontsize=14)

        plt.grid(True)  # 격자 무늬 추가
        ax.tick_params(axis='both', labelsize=12)  # x축과 y축 값의 글씨 크기 설정

        # ax.set_facecolor('skyblue')  # 배경을 파란색으로 설정

        # 범례 추가
        plt.legend(loc='upper center', fontsize=14)

        # 축 범위 지정
        plt.ylim([0, 8])

        # plt.title('Loss landscape perturbed based on top Hessian eigenvector')

        directory = self.args.experiment_name.replace('../', '')
        directory = 'landscape_image/' + directory + "/2d/"

        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' + directory)

        plt.savefig(directory + title + '.png')


    def save_landscape_3dimage(self, loss_list, loss_list2 , title):

        fig = plt.figure()
        landscape = fig.gca(projection='3d')
        landscape.plot_trisurf(loss_list[:, 0], loss_list[:, 1], loss_list[:, 2], alpha=0.8, cmap='viridis')
        landscape.plot_trisurf(loss_list2[:, 0], loss_list2[:, 1], loss_list2[:, 2], alpha=0.8, cmap='hot')

        landscape.set_title('Loss Landscape')
        landscape.set_xlabel('ε_1')
        landscape.set_ylabel('ε_2')
        landscape.set_zlabel('Loss')

        # z_min = min(min(loss_list[:,2]),min(loss_list2[:,2]))

        # landscape.set_zlim(z_min, z_min+13)
        landscape.view_init(elev=30, azim=45)
        landscape.dist = 6

        #plt.show()

        directory = self.args.experiment_name.replace('../', '')
        directory = 'landscape_image/' + directory + "/3d/"

        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' + directory)

        plt.savefig(directory + title + '.png')









