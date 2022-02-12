import torch
import torch.nn as nn
import torch.nn.functional as F

USE_ALEX = True


class ClassificationNetwork(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(ClassificationNetwork, self).__init__()

        class_network = 'AlexNet' if USE_ALEX else 'ResNet'
        print(f'Using {class_network} for classification task.')
        
        if USE_ALEX:
            # 1 x 53 x 53
            out_channels = 8
            self.conv_rel_max1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=2, padding=2),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
            )
            # 8 x 13 x 13

            in_channels = out_channels
            out_channels = 16
            self.conv_rel_max2 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
            )
            # 16 x 6 x 6

            in_channels = out_channels
            out_channels = 32
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(),
            )
            # 64 x 6 x 6

            in_channels = out_channels
            out_channels = 64
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(),
            )
            # 64 x 6 x 6

            in_channels = out_channels
            out_channels = 64
            self.conv3 = nn.Sequential(
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

            in_channels = out_channels * 3 * 3
            out_channels = int(in_channels / 4)
            self.fc_rel1 = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU()
            )
            # 8192

            in_channels = out_channels
            out_channels = n_classes
            self.fc1 = nn.Linear(in_channels, out_channels)
            # 3

            self.sigmoid = nn.Sigmoid()


        else:

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

            self.res_conv0 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(num_features=64),
            )

            in_channels = out_channels
            out_channels = 64
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(num_features=out_channels),
            )
            self.relu2 = nn.ReLU()
            # 64 x 13 x 13

            in_channels = out_channels
            out_channels = 128
            self.conv3 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(),
            )
            # 128 x 13 x 13I

            self.res_conv2 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(num_features=64),
            )

            in_channels = out_channels
            out_channels = 64
            self.conv4 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(num_features=out_channels),
            )
            self.relu4 = nn.ReLU()
            # 64 x 7 x 7

            in_channels = out_channels
            out_channels = 32
            self.conv5 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(),
            )
            # 32 x 7 x 7

            self.res_conv4 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=16),
            )

            in_channels = out_channels
            out_channels = 16
            self.conv6 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=out_channels),
            )
            self.relu6 = nn.ReLU()
            # 16 x 7 x 7

            self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
            # 16 x 1 x 1

            self.fc = nn.Linear(in_features=(out_channels*1*1), out_features=n_classes)
            self.sigmoid1 = nn.Sigmoid()


    def forward(self, x):

        if USE_ALEX:
            # Get characteristics
            x = self.conv_rel_max1(x)
            x = self.conv_rel_max2(x)

            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)

            x = self.conv_rel_max3(x)

            # Flatten
            x = x.view(x.shape[0], -1)

            # Identify classes
            x = self.fc_rel1(x)
            x = self.fc1(x)
            x = self.sigmoid(x)

            output = x

        else:
            x = self.conv0(x)
            x = self.max_pool(x)
            x1 = x

            x = self.conv1(x)
            x = self.conv2(x)
            x1 = self.res_conv0(x1)
            x = x + x1
            x2 = self.relu2(x)

            x = self.conv3(x2)
            x = self.conv4(x)
            x2 = self.res_conv2(x2)
            x = x + x2
            x3 = self.relu4(x)

            x = self.conv5(x3)
            x = self.conv6(x)
            x3 = self.res_conv4(x3)
            x = x + x3
            x4 = self.relu6(x)

            x = self.avg_pool(x4)

            x = x.view(x.shape[0], -1)
            x = self.fc(x)
            x = self.sigmoid1(x)

            output = x

        return output

