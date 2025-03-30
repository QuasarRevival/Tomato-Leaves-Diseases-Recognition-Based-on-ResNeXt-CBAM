import torch.nn as nn
from init_param import initialize_weights


# VGG16深神经网络
class VGG(nn.Module):
    def __init__(self, num_classes=10, init_weight=True):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            # Block 1: 2层Conv3-64
            # input = (N, C=3, H=256, W=256)
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 2层Conv3-128
            # input = (N, C=64, H=128, W=128)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 3层Conv3-256
            # input = (N, C=128, H=64, W=64)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 3层Conv3-512
            # input = (N, C=256, H=32, W=32)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5: 3层Conv3-512
            # input = (N, C=512, H=16, W=16)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 自适应池化层
        # input = (N, C=512, H=8, W=8)
        # self.average_pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        # 分类层（全连接）
        # input = (N, C=512, H=7, W=7)
        self.full_connection_1 = nn.Linear(512 * 4 * 4, 4096, bias=True)
        self.non_linear_1 = nn.ReLU(inplace=True)
        self.dropout_1 = nn.Dropout(p=0.5)
        self.full_connection_2 = nn.Linear(4096, 4096, bias=True)
        self.non_linear_2 = nn.ReLU(inplace=True)
        self.dropout_2 = nn.Dropout(p=0.5)
        self.full_connection_3 = nn.Linear(4096, num_classes, bias=True)

        if init_weight:
            initialize_weights(self)

    def forward(self, x):
        x = self.features(x)
        # x = self.average_pool(x)
        x = x.view(x.size(0), -1)

        x = self.full_connection_1(x)
        x = self.non_linear_1(x)
        x = self.dropout_1(x)
        x = self.full_connection_2(x)
        x = self.non_linear_2(x)
        x = self.dropout_2(x)
        x = self.full_connection_3(x)
        return x
