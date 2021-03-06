import torch.nn as nn
import torch

class UNet(nn.Module):
    def __init__(self, input_channels, n_classes):
        super(UNet, self).__init__()
        # ------------------------ Laboratoire 2 - Question 4 - Début de la section à compléter ------------------------
        self.hidden = 32  # ???

        # Down 1
        self.conv_1_1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.relu_1_1 = nn.ReLU()
        self.conv_1_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.relu_1_2 = nn.ReLU()

        # Down 2
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.relu_2_1 = nn.ReLU()
        self.conv_2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.relu_2_2 = nn.ReLU()

        # Down 3
        self.maxpool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.relu_3_1 = nn.ReLU()
        self.conv_3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.relu_3_2 = nn.ReLU()

        # Down 4
        self.maxpool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_4_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.relu_4_1 = nn.ReLU()
        self.conv_4_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.relu_4_2 = nn.ReLU()

        # Down 5
        self.maxpool_5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_5_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.relu_5_1 = nn.ReLU()
        self.conv_5_2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.relu_5_2 = nn.ReLU()

        # Up 6
        self.upsample_6 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2)
        self.conv_6_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.relu_6_1 = nn.ReLU()
        self.conv_6_2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.relu_6_2 = nn.ReLU()

        # Up 7
        self.upsample_7 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, padding=0, stride=2)
        self.conv_7_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.relu_7_1 = nn.ReLU()
        self.conv_7_2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.relu_7_2 = nn.ReLU()

        # Up 8
        self.upsample_8 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, padding=0, stride=2)
        self.conv_8_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.relu_8_1 = nn.ReLU()
        self.conv_8_2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.relu_8_2 = nn.ReLU()

        # Up 9
        self.upsample_9 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, padding=0, stride=2)
        self.conv_9_1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.relu_9_1 = nn.ReLU()
        self.conv_9_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.relu_9_2 = nn.ReLU()

        self.output_conv = nn.Conv2d(in_channels=self.hidden, out_channels=n_classes, kernel_size=1)

    def forward(self, x):

        # Down 1
        output = self.conv_1_1(x)
        output = self.relu_1_1(output)
        output = self.conv_1_2(output)
        output = self.relu_1_2(output)
        concat_1 = output.clone()

        # Down 2
        output = self.maxpool_2(output)
        output = self.conv_2_1(output)
        output = self.relu_2_1(output)
        output = self.conv_2_2(output)
        output = self.relu_2_2(output)
        concat_2 = output.clone()

        # Down 3
        output = self.maxpool_3(output)
        output = self.conv_3_1(output)
        output = self.relu_3_1(output)
        output = self.conv_3_2(output)
        output = self.relu_3_2(output)
        concat_3 = output.clone()

        # Down 4
        output = self.maxpool_4(output)
        output = self.conv_4_1(output)
        output = self.relu_4_1(output)
        output = self.conv_4_2(output)
        output = self.relu_4_2(output)
        concat_4 = output.clone()

        # Down 5
        output = self.maxpool_5(output)
        output = self.conv_5_1(output)
        output = self.relu_5_1(output)
        output = self.conv_5_2(output)
        output = self.relu_5_2(output)

        # Up 6
        output = self.upsample_6(output)
        output = torch.cat((output, concat_4), dim=1)
        output = self.conv_6_1(output)
        output = self.relu_6_1(output)
        output = self.conv_6_2(output)
        output = self.relu_6_2(output)

        # Up 7
        output = self.upsample_7(output)
        output = torch.cat((output, concat_3), dim=1)
        output = self.conv_7_1(output)
        output = self.relu_7_1(output)
        output = self.conv_7_2(output)
        output = self.relu_7_2(output)

        # Up 8
        output = self.upsample_8(output)
        output = torch.cat((output, concat_2), dim=1)
        output = self.conv_8_1(output)
        output = self.relu_8_1(output)
        output = self.conv_8_2(output)
        output = self.relu_8_2(output)

        # Up 9
        output = self.upsample_9(output)
        output = torch.cat((output, concat_1), dim=1)
        output = self.conv_9_1(output)
        output = self.relu_9_1(output)
        output = self.conv_9_2(output)
        output = self.relu_9_2(output)
        output = self.output_conv(output)

        # Out
        out = output

        return out
        # ------------------------ Laboratoire 2 - Question 4 - Fin de la section à compléter --------------------------
