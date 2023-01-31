import torch
import numpy
from sklearn.model_selection import KFold


class WPCValidator:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def generate_kfold(self):
        self.kf = KFold(
            n_splits=5, shuffle=True, random_state=42
        )  # TODO: test 10 splits

    def get_split(self, input_x, input_y, indexes):
        pass

# load  and preprocess training data
x_train = torch.load("data/x_tensor_WPC.pt")
x_train = numpy.array([[v.cpu().detach().numpy() for v in x] for x in x_train])
y_train = numpy.load("data/y_tensor_WPC.pt")


# kfold initialization and preparation
kf = KFold(n_splits=5, shuffle=True, random_state=42)


def get_split(input_x, input_y, indexes):
    output_x, output_y = [], []
    for index in indexes:
        output_x.append(input_x[index])
        output_y.append(input_y[index])


# getter functions for models

# load and preprocess test data

# training loop

# create and display result dataframes
