import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms

@DeprecationWarning
def get_individual_jacobian(model, input_image):
    '''

    :param output_embedding:
    :return:

    set_detect_anomaly(True) is used to explicitly raise an error with a stack trace to easier debug which operation might have created the invalid values.
     Without setting this global flag, the invalid values would just be created and the training might be broken (e.g. if you update any parameter to NaN).


    '''
    torch.autograd.set_detect_anomaly(True)
    model.zero_grad()

    input_image.requires_grad = True
    output_embedding = model(input_image)

    __output_embedding = torch.ones_like(output_embedding)

    # do backpropagation to collect the gradient which will work as jacobian matrix for input_image.
    output_embedding.backward(__output_embedding)
    # detach the input gradient from the computational graph.
    jacob = input_image.grad.detach()
    return jacob


@DeprecationWarning
def seperate_perclass_jacobians(jacob, labels=None):
    '''

    :param jacob:
    :param labels:
    :param classes:
    :return:
    '''

    temp = 1e-5
    perclass_jacobians = {}

    for i, label in enumerate(labels):
        if label in perclass_jacobians:
            # each image in the batch is getting stacked against its corresponding labels
            perclass_jacobians[label] = np.vstack((perclass_jacobians[label],jacob[i]))
        else:
            # if it is the first time then just put the matrix in the label.
            perclass_jacobians[label] = jacob[i]
    return perclass_jacobians






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





def get_batch_jacobian(encoder, projection_head, x):
    torch.autograd.set_detect_anomaly(True)
    encoder.zero_grad()
    projection_head.zero_grad()

    x.requires_grad_(True)

    h = encoder(x)
    z = projection_head(h)
    like_z = torch.ones_like(z)
    z.backward(like_z)
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



