import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionNetwork(nn.Module):
    def __init__(self, input_channels, n_params):
        super(DetectionNetwork, self).__init__()

        # À compléter
        raise NotImplementedError()

    def forward(self, x):
        
        # À compléter
        output = None

        return output


def FixedPredictorDetectionNetworkLoss():
    raise NotImplementedError()

def DetectionNetworkLoss():
    raise NotImplementedError()