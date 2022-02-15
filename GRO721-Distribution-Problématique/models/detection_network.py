import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
import numpy as np

N_BOITES = 4
WIDTH = 53

class DetectionNetwork(nn.Module):
    def __init__(self, in_channels, n_params):
        super(DetectionNetwork, self).__init__()

        self.n_params = n_params
        # 3, 53, 53
        pass



    def forward(self, x):
        output = x
        return output


def FixedPredictorDetectionNetworkLoss():
    pass


def DetectionNetworkLoss(prediction, target):
    pass
