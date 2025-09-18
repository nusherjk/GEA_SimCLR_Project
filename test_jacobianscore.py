import json

import torch

from load_cifer10 import load_cifar10
from models.simclr import SimCLR
from torch import nn
import numpy as np

import torchvision
import torchvision.transforms as transforms

device = 'cpu'




def load_cifar10_batch(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(224),  # ResNet expects 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = torchvision.datasets.CIFAR10(root="./data", train=True,
                                           download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True)
    return next(iter(loader))





def get_batch_jacobian(net, x):
    torch.autograd.set_detect_anomaly(True)
    net.zero_grad()

    x.requires_grad_(True)

    y = net(x)
    like_y = torch.ones_like(y)
    y.backward(like_y)
    jacob = x.grad.detach()
    # print(jacob)

    return jacob#, grad


def eval_score_perclass(jacob, labels=None, n_classes=10):
    '''

    :param jacob: (32 jacobian matrix of input images in the model
    :param labels:  y labels of each class (total 10)
    :param n_classes: 10 for cifer-10
    :return:
    '''
    #  Set temperature
    k = 1e-5
    #n_classes = len(np.unique(labels))
    # perclass jacobian sets
    per_class={}
    for i, label in enumerate(labels):
        # print(label)
        if label in per_class:
            # each image in the batch is getting stacked against its corresponding labels
            per_class[label] = np.vstack((per_class[label],jacob[i]))
        else:
            # if it is the first time then just put the matrix in the label.
            per_class[label] = jacob[i]


    # Calculating M_k correlation matrix for individual classes
    ind_corr_matrix_score = {}
    for c in per_class.keys():
        # print(c)
        s = 0
        try:
            corrs = np.corrcoef(per_class[c])

            s = np.sum(np.log(abs(corrs)+k))#/len(corrs)
            if n_classes > 100:
                s /= len(corrs)
        except: # defensive programming
            continue

        ind_corr_matrix_score[c] = s

    # per class-corr matrix A and B
    score = 0
    ind_corr_matrix_score_keys = ind_corr_matrix_score.keys()
    if n_classes <= 100:

        for c in ind_corr_matrix_score_keys:
            # B)
            # j = np.absolute(ind_corr_matrix_score[c])
            # print("{} for {}".format(j, c))
            score = score + np.absolute(ind_corr_matrix_score[c])
    else:
        for c in ind_corr_matrix_score_keys:
            # A)
            for cj in ind_corr_matrix_score_keys:
                score = score + np.absolute(ind_corr_matrix_score[c]-ind_corr_matrix_score[cj])

        # should divide by number of classes seen
        score /= len(ind_corr_matrix_score_keys)

    return score



def test_jacobian(mlp_arch):




    print(mlp_arch)
    # create Simclr architecture

    model = SimCLR(mlp_arch)
    # l = [module for module in model.modules() if not isinstance(module, nn.Sequential)]
    #
    # print(l)
    # train_loader, test_loader = load_cifar10()
    images, y = load_cifar10_batch()
    y = y.cpu().tolist()
    # print(images.shape)
    # data_iterator = iter(train_loader)
    # x, target = next(data_iterator)
    # x, target = x.to(device), target.cpu().tolist()

    jacobs_batch = get_batch_jacobian(model, images)
    # print(jacobs_batch.shape)
    ## flatten and 2dfy the batch
    jacobs_batch = jacobs_batch.reshape(jacobs_batch.size(0), -1).cpu().tolist()

    try:
        s = eval_score_perclass(jacobs_batch, y)

    except Exception as e:
        print(e)
        s = np.nan


    print(s)


if __name__ == '__main__':
    with open('utils/mlp_bench.json') as f:
        mlp_bench = json.load(f)
    #
    # # print(len(mlp_bench))
    rando_1 = mlp_bench[2985]
    test_jacobian(rando_1)
    rando_2 = mlp_bench[2985900]
    test_jacobian(rando_2)
    rando_3 = mlp_bench[298]
    test_jacobian(rando_3)
    rando_4 = mlp_bench[854]
    test_jacobian(rando_4)
    rando_5 = mlp_bench[9594]
    test_jacobian(rando_5)

    # s = [{'in': 512, 'out': 64, 'activation': 'relu'}]
    # test_jacobian(s)
    # for arch in mlp_bench:
    #     test_jacobian(arch)
    # rando_arch = mlp_bench[3]





