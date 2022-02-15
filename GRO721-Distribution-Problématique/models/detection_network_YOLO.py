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
        out_channels = 64
        self.conv_rel_max1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
        )
        # 64, 26, 26

        in_channels = out_channels
        out_channels = 32
        self.conv_rel1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
        )
        # 32, 26, 26

        in_channels = out_channels
        out_channels = 64
        self.conv_rel2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
        )
        # 64, 26, 26

        in_channels = out_channels
        out_channels = 32
        self.conv_rel3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
        )
        # 32, 26, 26

        in_channels = out_channels
        out_channels = 64
        self.conv_rel4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
        )
        # 64, 26, 26

        in_channels = out_channels
        out_channels = 256
        self.conv_rel_max2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
        )
        # 256, 13, 13

        in_channels = out_channels
        out_channels = 128
        self.conv_rel5 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
        )
        # 128, 13, 13

        in_channels = out_channels
        out_channels = 128
        self.conv_rel6 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
        )
        # 128, 7, 7

        in_channels = out_channels
        out_channels = 128
        self.conv_rel7 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
        )
        #128, 7, 7

        in_channels = out_channels
        out_channels = 128
        self.conv_rel8 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
        )
        # 128, 7, 7

        in_channels = out_channels
        out_channels = 128
        self.conv_rel9 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
        )
        # 128, 7, 7

        in_channels = out_channels
        out_channels = (n_params * 3) * N_BOITES * N_BOITES
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )
        # 144, 1, 1

        # Resize in forward
        self.sigmoid1 = nn.Sigmoid()
        # 9, 4, 4
        # [x, y, s, 0, 0, 0]


    def forward(self, x):

        # out = F.pad(x, (1, 1))      # pad last dim by 0 on each side

        x = self.conv_rel_max1(x)
        x = self.conv_rel1(x)
        x = self.conv_rel2(x)
        x = self.conv_rel3(x)
        x = self.conv_rel4(x)

        x = self.conv_rel_max2(x)
        x = self.conv_rel5(x)
        x = self.conv_rel6(x)
        x = self.conv_rel7(x)
        x = self.conv_rel8(x)
        x = self.conv_rel9(x)

        x = self.conv1(x)
        x = x.view(9, 4, 4)
        x = self.sigmoid1(x)

        # TODO: NMS

        # TODO: FIX PREDICTION FORMAT

        output = x

        return output

    @staticmethod
    def get_BBs(prediction):
        """
        :param prediction:
        (9 * 4 * 4)

        [
        [],
        [...].
        ...x9
        ]

        ...
        (4 x 4 x 9)
        box coordinates: scaled to total image size
        :return:

        global bboxes... after non-max suppression
        """

        # Resize to have the information separated by boxelet
        # Information (size: 9): [pres, x_c, y_c, s, [4:8]: class multi-hot]]
        prediction = prediction.reshape(N_BOITES, N_BOITES, -1)

        CELL_WIDTH = WIDTH//N_BOITES

        # Convert predicted cell boxes to big boxes coordinates
        all_bboxes = []
        for i, row_cellboxes in enumerate(prediction):
            base_y = CELL_WIDTH * i
            for j, cellbox_pred in enumerate(row_cellboxes):
                # [pres_confiance, x_c, y_c, s, [4:8]: class multi-hot]]
                base_x = CELL_WIDTH * j

                x_global = int(base_x + cellbox_pred[1] * CELL_WIDTH)
                y_global = int(base_y + cellbox_pred[2] * CELL_WIDTH)

                converted_pred = cellbox_pred.clone()  # TODO: Need to clone?
                converted_pred[1] = x_global
                converted_pred[2] = y_global
                all_bboxes.append(converted_pred)

        # TODO: apply non-max suppression to only keep relevant bboxes
        # all_bboxes...




def FixedPredictorDetectionNetworkLoss():
    pass


def compute_iou(boxA, boxB):

    x1, y1, s1 = boxA
    x2, y2, s2 = boxB

    # determine the (x, y) coordinates of the intersection rectangle
    x0_inter = max(x1, x2)
    y0_inter = max(y1, y2)
    x1_inter = min(x1 + s1, x2 + s2)
    y1_inter = min(y1 + s1, y2 + s2)

    # compute the area of intersection rectangle
    width_inter = max(0, x1_inter - x0_inter + 1)
    height_inter = max(0, y1_inter - y0_inter + 1)
    inter_area = width_inter * height_inter

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = s1 * s1
    boxBArea = s2 * s2

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = inter_area / float(boxAArea + boxBArea - inter_area)

    return iou


def DetectionNetworkLoss(prediction, target):

    """
    prediction
    [
        9 * 4 * 4
    ]

    target:
    [
        [0, x_c, y_c, s, triangle]
        [0, x_c, y_c, s, cercle]
        [1, x_c, y_c, s, cross]
    ] * N
    """

    loss_total = 0
    for pred, tar in zip(prediction, target):
        l_box = 0
        l_conf_obj = 0
        l_conf_no_obj = 0
        l_class = 0

        pred_BBs = DetectionNetwork.get_BBs(prediction)

        boxlet_target_array = np.zeros((N_BOITES,N_BOITES))

        # TODO: This might have to be two for loops: cannot guarantee that len(pred_BBS) == len(tar)
        for pred_shape, tar_shape in zip(pred_BBs, tar):
            if tar_shape[0]:
                # Boites englobantes
                x_diff = (tar_shape[1] - pred_shape[1]) ** 2
                y_diff = (tar_shape[2] - pred_shape[2]) ** 2
                s_diff = (torch.sqrt(tar_shape[3]) - torch.sqrt(pred_shape[3])) ** 2
                l_box += x_diff + y_diff + 2 * s_diff

                iou_shape = compute_iou(pred_shape[1:-1], tar_shape[1:-1])
                l_conf_obj += nn.BCELoss(pred_shape[0], iou_shape)
            else:
                l_conf_no_obj += nn.BCELoss(pred_shape[0], 0)

        l_class = nn.BCELoss(pred[1], boxlet_target_array)  # TODO: Does this work?? Format of data??

        loss_total += 5 * l_box + l_conf_obj + 0.5 * l_conf_no_obj + l_class

    return loss_total
