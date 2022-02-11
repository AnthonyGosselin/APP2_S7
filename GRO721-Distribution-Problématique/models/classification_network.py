import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationNetwork(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(ClassificationNetwork, self).__init__()

        # 1 x 53 x 53
        out_channels = 16
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )
        # 16 x 53 x 53
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 16 x 26 x 26

        in_channels = out_channels
        out_channels = 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )
        # 32 x 26 x 26

        in_channels = out_channels
        out_channels = 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )
        # 64 x 13 x 13

        in_channels = out_channels
        out_channels = 128
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )
        # 128 x 13 x 13

        in_channels = out_channels
        out_channels = 64
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )
        # 64 x 7 x 7

        in_channels = out_channels
        out_channels = 32
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )
        # 32 x 7 x 7

        in_channels = out_channels
        out_channels = 16
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )
        # 16 x 7 x 7

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        # 16 x 1 x 1

        self.fc = nn.Linear(in_features=(out_channels*1*1), out_features=n_classes)
        self.sigmoid1 = nn.Sigmoid()


    def forward(self, x):
        
        # À compléter
        output = None

        return output
