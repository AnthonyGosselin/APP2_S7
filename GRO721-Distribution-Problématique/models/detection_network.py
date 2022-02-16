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
        out_channels = 32
        self.conv_rel_max1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=1, padding=2),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
        )
        # 16 x 26 x 26

        in_channels = out_channels
        out_channels = 128
        self.conv_rel_max2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
        )
        # 32 x 13 x 13

        in_channels = out_channels
        out_channels = 128
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
        )
        # 64 x 6 x 6

        in_channels = out_channels
        out_channels = 128
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
        )
        # 128 x 3 x 3

        # FORK into two linear layers
        in_channels = out_channels

        # Head for boxes
        out_channels = 16
        self.fc_conv_box1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
        # Output 1: 16 x 3 x 3

        in_channels = out_channels
        out_channels = 3
        self.fc_conv_box2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
        )
        # Output 1: 3 x 2 x 2
        # Reshape: 3 x 4 (Boxes)
        self.sigmoid_box = nn.Sigmoid()

        # END: box branch ----------

        # Head for classes
        in_channels = 128
        out_channels = 16
        self.fc_conv_class1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
        )
        # Output 2: 16 x 3 x 3

        in_channels = out_channels
        out_channels = 3
        self.fc_conv_class2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 5), stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
        )
        # Output 2: 3 x 3 x 1
        # Reshape: 3 x 3
        self.sigmoid_class = nn.Sigmoid()

        # END: class branch -----------

        # Concat output from forks: 3 x 7
        # [confidence, x, y, scale, one-hot(3)] for each class (21 total outputs)



    def forward(self, x):
        # Get characteristics
        x = self.conv_rel_max1(x)
        x = self.conv_rel_max2(x)

        x = self.conv1(x)
        x = self.conv2(x)

        # Fork
        out1 = self.fc_conv_box1(x)
        out1 = self.fc_conv_box2(out1)
        out1 = out1.view(-1, 3, 4)
        out1 = self.sigmoid_box(out1)

        out2 = self.fc_conv_class1(x)
        out2 = self.fc_conv_class2(out2)
        out2 = out2.view(-1, 3, 3)
        out2 = self.sigmoid_class(out2)

        # Concat output
        output = torch.cat((out1, out2), dim=2)

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
    loss_total = 0

    # [1,2,5,6,0] [1,3,4,5,2] [0,0,0,0,0]

    # Loop through batch
    for pred, tar in zip(prediction, target):
        l_box = 0
        l_conf_obj = 0
        l_conf_no_obj = 0
        l_class = 0

        # Sort target
        sorted_target = torch.zeros((3, 5))
        for t in tar:
            if t[0]:
                class_ind = t[4]
                sorted_target[int(class_ind.item())] = t

        # Loop through shapes
        for pred_shape, tar_shape in zip(pred, sorted_target):
            if tar_shape[0]:
                x_diff = (tar_shape[1] - pred_shape[1]) ** 2
                y_diff = (tar_shape[2] - pred_shape[2]) ** 2
                s_diff = (torch.sqrt(tar_shape[3]) - torch.sqrt(pred_shape[3])) ** 2
                l_box += x_diff + y_diff + 2 * s_diff

                iou_shape = detection_intersection_over_union(pred_shape[1:4], tar_shape[1:4])
                indiv_l_conf_obj = F.binary_cross_entropy(pred_shape[0], iou_shape)
                l_conf_obj += indiv_l_conf_obj
            else:
                l_conf_no_obj += F.binary_cross_entropy(pred_shape[0], tar_shape[0])

        t = sorted_target[:, 4].long()
        l_class = F.cross_entropy(pred[:, 4:], t)

        loss_total += 10 * l_box + 0.5 * l_conf_obj + 0.5 * l_conf_no_obj + l_class

    return loss_total
