import torch.nn as nn
import torch.nn.init as init
from init_param import initialize_weights


# ResNet残差网络模型
# 瓶颈类定义
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1, downsample=None, init_weight=True):
        super(Bottleneck, self).__init__()

        # 降低维度层
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)

        # 提取特征层
        # 注意此处卷积核移动步长stride
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        # 恢复维度层
        self.conv3 = nn.Conv2d(out_planes, out_planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

        if init_weight:
            initialize_weights(self)
            # 残差层最后一层可以考虑使用较小的值进行初始化
            init.normal_(self.conv3.weight, mean=0, std=0.01)

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

        # 若有下采样，则下采样
        if self.downsample is not None:
            residual = self.downsample(x)

        # return_value(x) = last_out(x) + x
        # H(x) = F(x) + x
        out += residual
        out = self.relu(out)
        return out


# ResNet50定义
class ResNet(nn.Module):
    def __init__(self, num_classes, init_weight=True):
        super(ResNet, self).__init__()

        # 感知层，卷积核边长为7
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 主体结构层
        self.layer1 = self._make_layer(64, 64, 3, init_weight=init_weight)
        self.layer2 = self._make_layer(256, 128, 4, stride=2, init_weight=init_weight)
        self.layer3 = self._make_layer(512, 256, 6, stride=2, init_weight=init_weight)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2, init_weight=init_weight)

        # 自适应池化层，替代传统的复杂全连接层，减少参数数量，避免过拟合
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)

        # 分类器，全连接层
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
        # 分类器待替换：
        # 注意LayerNorm和BatchNorm的区别：
        # LayerNorm：对单个样本的所有通道进行正则化
        # BatchNorm：对所有样本的单个通道进行正则化
        # 可以尝试使用GroupNorm，是LayerNorm的简化版，将单个样本的通道进行分组，每一组内进行正则化，性能更加稳定
        '''
        self.classifier = nn.Sequential(
            nn.Linear(512 * Bottleneck.expansion, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes)
        )
        '''

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
        layers = [Bottleneck(in_planes, out_planes, stride, downsample, init_weight)]
        in_planes = out_planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(in_planes, out_planes, init_weight=init_weight))
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pooling(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pooling(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
