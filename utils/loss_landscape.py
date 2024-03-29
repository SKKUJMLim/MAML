import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
import copy
from mypyhessian import my_hessian
import torch.nn.functional as F

import os


class landscape(nn.Module):
    def __init__(self, model, args):
        super(landscape, self).__init__()

        self.model = model
        self.args = args

    def get_params(self, model_orig, model_perb, direction, alpha):
        i=0
        for m_orig, m_perb in zip(model_orig.parameters(), model_perb.parameters()):
            if m_orig.requires_grad:
                m_perb.data = m_orig.data + alpha * direction[i]
                i = i+1
        return model_perb

    def save_landscape_3dimage(self, loss_list, title):

        loss_list = np.array(loss_list)

        fig = plt.figure()
        landscape = fig.gca(projection='3d')
        #landscape.plot_trisurf(loss_list[:, 0], loss_list[:, 1], loss_list[:, 2], alpha=0.8, cmap='viridis')
        landscape.plot_trisurf(loss_list[:, 0], loss_list[:, 1], loss_list[:, 2], alpha=0.8, cmap='hot')
        # cmap=cm.autumn, #cmamp = 'hot')

        landscape.set_title('Loss Landscape')
        landscape.set_xlabel('ε_1')
        landscape.set_ylabel('ε_2')
        landscape.set_zlabel('Loss')

        landscape.view_init(elev=15, azim=75)
        #landscape.view_init(elev=30, azim=45)
        landscape.dist = 6


        directory = self.args.experiment_name.replace('../', '')
        directory = 'landscape_image/' + directory + "/3d/"

        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' + directory)

        plt.savefig(directory + title + '.png')


    def show(self, inputs, targets, title):

        model = self.model.cuda()
        inputs, targets = inputs.cuda(), targets.cuda()

        hessian_comp = my_hessian.my_hessian(model, data=(inputs, targets), cuda=True)

        # get the top eigenvector
        top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)

        # lambda is a small scalar that we use to perturb the model parameters along the eigenvectors
        lams1 = np.linspace(-0.5, 0.5, 21).astype(np.float32)
        lams2 = np.linspace(-0.5, 0.5, 21).astype(np.float32)

        model_perb1 = copy.deepcopy(model)
        model_perb1.eval()
        model_perb1.cuda()

        model_perb2 = copy.deepcopy(model)
        model_perb2.eval()
        model_perb2.cuda()

        loss_list = []

        for lam1 in lams1:
            for lam2 in lams2:
                model_perb1 = self.get_params(model, model_perb1, top_eigenvector[0], lam1)
                model_perb2 = self.get_params(model_perb1, model_perb2, top_eigenvector[1], lam2)
                preds = model_perb2.forward(x=inputs, num_step=5)
                loss = F.cross_entropy(input=preds, target=targets)

                loss_list.append((lam1, lam2, loss.item()))

        self.save_landscape_3dimage(loss_list, title)


    def save_landscape_2dimage(self, lams, loss_list, title):

        fig, ax = plt.subplots()
        plt.plot(lams, loss_list)
        plt.ylabel('Loss')
        plt.xlabel('Perturbation')

        # 축 범위 지정
        plt.ylim([0, 15])

        plt.title('Loss landscape perturbed based on top Hessian eigenvector')

        directory = self.args.experiment_name.replace('../', '')
        directory = 'landscape_image/' + directory + "/2d/"

        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' + directory)

        plt.savefig(directory + title + '.png')


    def show_2d(self, inputs, targets, title):
        model = self.model.cuda()
        inputs, targets = inputs.cuda(), targets.cuda()

        hessian_comp = my_hessian.my_hessian(model, data=(inputs, targets), cuda=True)
        top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()

        lams = np.linspace(-0.5, 0.5, 21).astype(np.float32)

        model_perb = copy.deepcopy(model)
        model_perb.eval()
        model_perb.cuda()

        loss_list = []

        for lam in lams:
            model_perb = self.get_params(model, model_perb, top_eigenvector[0], lam)
            preds, out_feature_dict = model_perb.forward(x=inputs, num_step=5)
            loss = F.cross_entropy(input=preds, target=targets)
            loss_list.append(loss.item())

        self.save_landscape_2dimage(lams, loss_list, title)







