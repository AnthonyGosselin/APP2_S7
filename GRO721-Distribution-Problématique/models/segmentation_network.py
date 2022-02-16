import torch
import torch.nn as nn
import torch.nn.functional as F

# --lr 5e-3


class SegmentationNetwork(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(SegmentationNetwork, self).__init__()

        # Down 1
        out_channels = 32
        self.conv_1_1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU()
        )
        self.conv_1_2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
        # 16 x 53 x 53

        # Down 2
        in_channels = out_channels
        out_channels = 64
        self.maxpool_2 = nn.MaxPool2d(kernel_size=3, stride=3)
        # 16 x 17 x 17
        self.conv_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
        self.conv_2_2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
        # 32 x 17 x 17

        # Down 3
        in_channels = out_channels
        out_channels = 94
        self.maxpool_3 = nn.MaxPool2d(kernel_size=3, stride=3)
        # 32 x 5 x 5
        self.conv_3_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
        self.conv_3_2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
        # 104 x 5 x 5

        # Down 4
        in_channels = out_channels
        out_channels = 2 * in_channels
        self.maxpool_4 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 104 x 2 x 2
        self.conv_4_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
        # 208 x 2 x 2
        self.conv_4_2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU()
        )
        # 104 x 2 x 2

        # Up 5
        # in_channels = in_channels
        out_channels = 64
        self.upsample_5 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels,
                                             output_padding=0, kernel_size=3, stride=2)
        # 104 x 5 x 5
        self.conv_5_1 = nn.Sequential(
            nn.Conv2d(in_channels=2*in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU()
        )
        self.conv_5_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
        # 32 x 5 x 5

        # Up 6
        in_channels = out_channels
        out_channels = 32
        self.upsample_6 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels,
                                             output_padding=2, kernel_size=3, stride=3)
        # 32 x 17 x 17
        self.conv_6_1 = nn.Sequential(
            nn.Conv2d(in_channels=2*in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU()
        )
        self.conv_6_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
        # 16 x 17 x 17

        # Up 7
        in_channels = out_channels
        out_channels = 32
        self.upsample_7 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels,
                                             output_padding=2, kernel_size=3, stride=3)
        # 16 x 53 x 53
        self.conv_7_1 = nn.Sequential(
            nn.Conv2d(in_channels=2 * in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU()
        )
        self.conv_7_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
        # 16 x 53 x 53

        self.output_conv = nn.Conv2d(in_channels=out_channels, out_channels=n_classes, kernel_size=1)
        # 4 x 53 x 53

    def forward(self, x):

        # Down 1
        output = self.conv_1_1(x)
        output = self.conv_1_2(output)
        concat_1 = output.clone()

        # Down 2
        output = self.maxpool_2(output)
        output = self.conv_2_1(output)
        output = self.conv_2_2(output)
        concat_2 = output.clone()

        # Down 3
        output = self.maxpool_3(output)
        output = self.conv_3_1(output)
        output = self.conv_3_2(output)
        concat_3 = output.clone()

        # Down 4
        output = self.maxpool_4(output)
        output = self.conv_4_1(output)
        output = self.conv_4_2(output)

        # Up 5
        output = self.upsample_5(output)
        output = torch.cat((output, concat_3), dim=1)
        output = self.conv_5_1(output)
        output = self.conv_5_2(output)

        # Up 6
        output = self.upsample_6(output)
        output = torch.cat((output, concat_2), dim=1)
        output = self.conv_6_1(output)
        output = self.conv_6_2(output)

        # Up 7
        output = self.upsample_7(output)
        output = torch.cat((output, concat_1), dim=1)
        output = self.conv_7_1(output)
        output = self.conv_7_2(output)

        output = self.output_conv(output)

        return output
