import torch.nn as nn
import CBAM


class FingerNet(nn.Module):
    def __init__(self, output):
        super(FingerNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.attention1 = CBAM.ChannelAttention(64)
        self.attention2 = CBAM.SpatialAttention()

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=384, kernel_size=3, padding=2, stride=2),
            nn.BatchNorm2d(384),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=512, kernel_size=3, padding=2),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.attention3 = CBAM.ChannelAttention(512)
        self.attention4 = CBAM.SpatialAttention()

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Dropout2d(0.4),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1152, 2048),
            nn.GELU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(2048, output)
        )

    def forward(self, x):
        x = self.layer1(x)
        # print(x.shape)
        x = self.attention1(x) * x
        x = self.attention2(x) * x
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.attention3(x) * x
        # print(x.shape)
        x = self.attention4(x) * x
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)
        x = x.reshape(x.size(0), -1)
        # print(x.shape)
        x = self.fc1(x)
        # print(x.shape)
        x = self.fc2(x)
        # print(x.shape)
        return nn.functional.log_softmax(x, dim=1)
