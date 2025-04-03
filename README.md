# Deep CNN: SSA-ResNeXt-CBAM

## How to work

- Create a folder named DeepCNN on your own computer
- Run these commands in your shell:
```shell
git init
git remote add https://github.com/QuasarRevival/DeepCNN.git
git pull origin master
```
**Don't forget to pull the latest repository every time you start coding!**
Then start you work on your local repository.

- When you finish your work, just simply type in these commands in your shell:
```shell
git add .
git commit -m"your commit message"
git push origin master
```
- Tips: how to add an appropriate commit message?
  - feat: add new features or functions
  - fix: fix bugs
  - refactor: other code modification except for feat. and fix.
  - docs: modify documentations
  - test: add test examples

**Remember: Don't push before carefully checking your modification on codes!**

## Datasets

- TomatoLeavesDataset:\
    https://www.kaggle.com/datasets/ashishmotwani/tomato/data
- PlantVillageTomatoLeavesDataset:
    - for images of healthy leaves and diseases except for powdery mildew:\
    https://www.kaggle.com/datasets/sunilgautam/dataset-of-tomato-leaves
    - for images of powdery mildew:\
    https://www.kaggle.com/datasets/joseenriquelopez/tomato-leaf-diseases

## Optimization Log

### Time: 2025.3.26

- Introduce parameter-initializing function of models

```python
import torch.nn as nn
import torch.nn.init as init


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            init.constant_(m.bias, 0)
```

For convolution layers and linear layers we choose Kaiming Normalization, 
since we use ReLU Non-linear Function in the models. It has been proved that Kaiming Normalization 
works well on models based on ReLU, and Xavier Normalization on Sigmoid.
But maybe using both of them in the same model can be an alternative.

- Result of the Day: \
Model: ResNet\
Test Accuracy: 98.706%\
Note: I don't think initialize the parameters of Bottleneck in ResNet is wrong, 
while the accuracy does decrease comparing to that in yesterday. I conjecture that there might be something 
with the Bottleneck initialization, i.e., the normalization of the last convolution layer of each Bottleneck.

### Time: 2025.3.27

- Use smaller kernel size in the perception layer of ResNet model
```python
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, num_classes, init_weight=True):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
```

The block can be replaced by:

```python
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, num_classes, init_weight=True):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
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
```

This will help to make the model deeper without adding too many parameters to it. 
With the network getting deeper, its ability of recognition will increase as well.
In addition, reducing the number of parameters also help to shrink the complexity 
of the network, reducing the risk of over-fitting at the same time.

- Optimize the classifier of ResNet model

```python
import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4
    # ...

class ResNet(nn.Module):
    def __init__(self, num_classes, init_weight=True):
        super(ResNet, self).__init__()
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
```

The classifier can be replaced by:

```python
import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4
    # ...

class ResNet(nn.Module):
    def __init__(self, num_classes, init_weight=True):
        super(ResNet, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512 * Bottleneck.expansion, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes)
        )
```

The usage of LayerNorm or GroupNorm may increase the stability of the model
when handling image classifying missions.

After altering the classifier, I found that maybe it's not a good idea to build a 
much more complicated classifier. But the optimization on the perception layer did work well.

- Result of the Day: \
Model: ResNet\
Test Accuracy: 97.445%\
Notes: The result may not be so reasonable, due to the increasing of epoch to 40 without any learning rate decay.

### Time: 2025.3.28

- No optimization to the model
- Result of the day: \
Model: ResNet\
Test Accuracy: 98.971%\
Notes: readjust the optimizer and learning rate scheduler.

### Time: 2025.3.30

- Add YOLOv8 model to the project and attempt to detect leaves from images 
and cut them from the original image. But the YOLOv8 model cannot do this work well. 
I guess that there might be some severe problems with the detection test dataset.
I found that images from the dataset all have leaves distinct from the background, 
which means that the model trained on this dataset won't do well to distinguish objects share 
similar color with the background. To solve the problem, we need to find images of leaves that might not 
be distinct from the background, and label them.

