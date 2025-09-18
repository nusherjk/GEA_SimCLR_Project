import torch
import numpy as np

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




