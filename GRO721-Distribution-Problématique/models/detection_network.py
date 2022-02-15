import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
import numpy as np
from metrics import detection_intersection_over_union

N_BOITES = 4
WIDTH = 53

class DetectionNetwork(nn.Module):
    def __init__(self, in_channels, n_params):
        super(DetectionNetwork, self).__init__()

        self.n_params = n_params

        # 1 x 53 x 53
        out_channels = 16
        self.conv_rel_max1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=2, padding=2),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
        )
        # 8 x 13 x 13

        in_channels = out_channels
        out_channels = 32
        self.conv_rel_max2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
        )
        # 16 x 6 x 6

        in_channels = out_channels
        out_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )
        # 64 x 6 x 6

        in_channels = out_channels
        out_channels = 128
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )
        # 64 x 6 x 6

        in_channels = out_channels
        out_channels = 64
        self.conv_rel_max3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
        )
        # 64 x 3 x 3

        # Rehshape:

        in_channels = out_channels * 3 * 3
        out_channels = int(in_channels / 16)
        self.fc_rel1 = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU()
        )
        # 36

        n_classes = 3
        in_channels = out_channels
        out_channels = (n_classes * n_params)  # Params: [confidence, x, y, scale, 0, 1, 0]
        self.fc1 = nn.Linear(in_channels, out_channels)
        # 3 x 7

        # NOTE: output
        # [confidence, x, y, scale, one-hot(3)] for each class (21 outputs)

        self.sigmoid = nn.Sigmoid()



    def forward(self, x):
        # Get characteristics
        x = self.conv_rel_max1(x)
        x = self.conv_rel_max2(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = self.conv_rel_max3(x)

        # Flatten
        x = x.view(x.shape[0], -1)

        # Get parameters for each class
        x = self.fc_rel1(x)
        x = self.fc1(x)
        x = self.sigmoid(x)

        output = x

        return output


def FixedPredictorDetectionNetworkLoss():
    pass


def DetectionNetworkLoss(prediction, target):
    """
    :param prediction:
    [
    [confidence, x, y, scale, 1, 0, 0],
    [confidence, x, y, scale, 0, 1, 0],
    [confidence, x, y, scale, 0, 0, 1],
    ] x batch_size
    (N, 3, 7)

    :param target:
    [
        [0, x_c, y_c, s, triangle]
        [0, x_c, y_c, s, cercle]
        [1, x_c, y_c, s, cross]
    ] * batch_size
    (N, 3, 5)

    :return: total loss for the batch
    """
    # lambda_xywh, lambda_conf_obj, lambda_conf_no_obj, lambda_classes = 1, 1, 1, 1  # TODO: adjust values
    loss_total = 0

    # Loop through batch
    # TODO: len(prediction) == len(target) ??
    for pred, tar in zip(prediction, target):
        l_box = 0
        l_conf_obj = 0
        l_conf_no_obj = 0
        l_class = 0

        # TODO: length match?
        for pred_shape, tar_shape in zip(pred, tar):
            if tar_shape[0]:
                x_diff = (tar_shape[1] - pred_shape[1]) ** 2
                y_diff = (tar_shape[2] - pred_shape[2]) ** 2
                s_diff = (torch.sqrt(tar_shape[3]) - torch.sqrt(pred_shape[3])) ** 2
                l_box += x_diff + y_diff + 2 * s_diff

                iou_shape = detection_intersection_over_union(pred_shape[1:], tar_shape[1:-1])
                l_conf_obj += nn.BCELoss(pred_shape[0], iou_shape)
            else:
                l_conf_no_obj += nn.BCELoss(pred_shape[0], 0)

            pred_classes = pred[4:]
            tar_classes = torch.zeros(3)
            tar_classes[tar_shape[4]] = 1
            l_class = nn.CrossEntropyLoss(pred_classes, tar_classes)

        loss_total += 5 * l_box + l_conf_obj + 0.5 * l_conf_no_obj + l_class

    # total_loss = 0
    # total_loss += lambda_xywh * l_xywh
    # total_loss += lambda_conf_obj * l_conf_obj
    # total_loss += lambda_conf_no_obj * l_conf_no_obj
    # total_loss += lambda_classes * l_classes

    return loss_total
