import torch as tc
import torch.nn as nn
from CNN_models.init_param import initialize_weights


# SEBlock模块定义
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# SpatialAttention模块定义
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = tc.mean(x, dim=1, keepdim=True)
        max_out, _ = tc.max(x, dim=1, keepdim=True)
        y = tc.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        y = self.sigmoid(y)
        return x * y


# ResNeXt残差网络定义
# 混合注意力机制运用，CBAM模块
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = SEBlock(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# 瓶颈类定义
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1,
                 cardinality=32, base_width=4, downsample=None, init_weight=True,
                 SE_reduction_ratio=16, SA_kernel_size=7):
        super().__init__()
        width = int(out_channels * (base_width / 64)) * cardinality
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        # 分组卷积（基数=cardinality，每组通道数=width//cardinality）
        self.conv2 = nn.Conv2d(
            width, width, kernel_size=3, stride=stride, padding=1,
            groups=cardinality, bias=False
        )
        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.cbam = CBAM(out_channels * self.expansion, SE_reduction_ratio, SA_kernel_size)
        self.relu = nn.ReLU(inplace=True)

        # 下采样（当输入输出维度不匹配时）
        self.downsample = downsample

        if init_weight:
            initialize_weights(self)

    def forward(self, x):
        # 学习残差
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.cbam(out)

        # 若有下采样，则下采样
        if self.downsample is not None:
            residual = self.downsample(x)

        # return_value(x) = last_out(x) + x
        # H(x) = F(x) + x
        out += residual
        out = self.relu(out)
        return out


# ResNeXt50-CBAM定义
class ResNeXt(nn.Module):
    def __init__(self, num_classes, init_weight=True, dropout_rate=0.5,
                 cardinality=32, base_width=4, SE_reduction_ratio=16, SA_kernel_size=7):
        super(ResNeXt, self).__init__()

        # hyperparameters
        self.cardinality = cardinality
        self.base_width = base_width
        self.SE_reduction_ratio = SE_reduction_ratio
        self.SA_kernel_size = SA_kernel_size

        # 感知层，卷积核边长为7
        '''
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        '''

        # 感知层替换：使用多个拥有较小尺寸卷积核的卷积层替代原先的大尺寸卷积核感知层
        self.perception = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=1)
        )

        # 主体结构层
        self.layer1 = self._make_layer(64, 64, 3, init_weight=init_weight)
        self.layer2 = self._make_layer(256, 128, 4, stride=2, init_weight=init_weight)
        self.layer3 = self._make_layer(512, 256, 6, stride=2, init_weight=init_weight)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2, init_weight=init_weight)

        # 自适应池化层，替代传统的复杂全连接层，减少参数数量，避免过拟合
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)

        # 分类器，全连接层

        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        if init_weight:
            initialize_weights(self)

    # 生成主体结构层：残差块
    def _make_layer(self, in_planes, out_planes, blocks, stride=1, init_weight=True):
        downsample = None

        # 创建下采样层：卷积核移动步长不为1，或扇入、扇出通道数不匹配时，加入下采样层
        # 下采样层位于每一个残差块的第一个瓶颈处
        # 下采样层会加载在跳跃连接处，对x直接起作用
        if stride != 1 or in_planes != out_planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes * Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes * Bottleneck.expansion)
            )

        # 第一个瓶颈有下采样层
        layers = [Bottleneck(in_planes, out_planes, stride, self.cardinality, self.base_width, downsample, init_weight,
                             SE_reduction_ratio=self.SE_reduction_ratio, SA_kernel_size=self.SA_kernel_size)]
        in_planes = out_planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(in_planes, out_planes, 1, self.cardinality, self.base_width, None,
                                     init_weight=init_weight, SE_reduction_ratio=self.SE_reduction_ratio,
                                     SA_kernel_size=self.SA_kernel_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        """

        x = self.perception(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pooling(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        x = self.fc(x)
        """

        x = self.classifier(x)
        """

        return x
