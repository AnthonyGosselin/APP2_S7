import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # ---------------------- Laboratoire 1 - Question 5 et 6 - Début de la section à compléter ---------------------
        # self.fc1 = nn.Linear(28 * 28, 10)

        # self.conv_fc = nn.Conv2d(1, 10, (28, 28))

        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(2)
        self.relu2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(7 * 7 * 2, 10)

        # ---------------------- Laboratoire 1 - Question 5 et 6 - Fin de la section à compléter -----------------------

    def forward(self, x):
        # ---------------------- Laboratoire 1 - Question 5 et 6 - Début de la section à compléter ---------------------
        # output = self.fc1(x.view(x.shape[0], -1))

        # output = self.conv_fc(x)
        # output = output.squeeze()

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.max_pool1(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.max_pool2(output)
        output = output.view(output.shape[0], -1)
        output = self.fc(output)

        # ---------------------- Laboratoire 1 - Question 5 et 6 - Fin de la section à compléter -----------------------
        return output
