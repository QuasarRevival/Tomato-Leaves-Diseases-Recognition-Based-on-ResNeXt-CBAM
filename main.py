import torch as tc
import torch.optim as optim
import torch.nn as nn
import random

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as scheduler

from model_VGG import VGG
from model_ResNet import ResNet


# GPU是否可用，设置随机数种子使得训练过程可复现
device = 'cuda' if tc.cuda.is_available() else 'cpu'
tc.manual_seed(415)
if device == 'cuda':
    tc.cuda.manual_seed_all(415)


# 训练集与验证集路径
train_path = './TomatoLeavesDatasets/train'
valid_path = './TomatoLeavesDatasets/valid'


# 定义种类字典
class_to_index = {
    'Bacterial_spot': 0,
    'Early_blight': 1,
    'Healthy': 2,
    'Late_blight': 3,
    'Leaf_Mold': 4,
    'Septoria_leaf_spot': 5,
    'Spider_mites Two-spotted_spider_mite': 6,
    'Target_spot': 7,
    'Tomato_mosaic_virus': 8,
    'Tomato_yellow_leaf_curl_virus': 9,
    'powdery_mildew': 10
}


# 设置超参数
image_size = (128, 128)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

learning_rate = 0.001
batch_size = 32
train_epoch = 25
l2_lambda = 0.00001


# 数据增强以及图像张量化
transform_train = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomRotation(degrees=(-15, 15)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
transform_valid = transforms.Compose([
    transforms.CenterCrop(size=(224, 224)),
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


# 通过路径加载数据
train_dataset = datasets.ImageFolder(
    root=train_path,
    transform=transform_train
)
valid_dataset = datasets.ImageFolder(
    root=valid_path,
    transform=transform_valid
)
train_dataset.class_to_idx = class_to_index
valid_dataset.class_to_idx = class_to_index


# 数据可训练化
train_data = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    pin_memory=True if device == 'cuda' else False
)

valid_data = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=False,
    pin_memory=True if device == 'cuda' else False
)


# 校验准确率
def evaluate_acc(test_loader, model):
    model.eval()
    test_correct = 0.0
    test_total = 0.0
    with tc.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            inputs, labels = images.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = tc.argmax(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    test_accuracy = test_correct / test_total
    return test_accuracy


# 训练模型
def train(model, train_loader, valid_loader, criterion, optimizer, schedule, epoch=25):
    print('Training starts.')
    model.train()
    total_batch = len(train_loader)
    for i in range(epoch):
        average_cost = 0
        train_accuracy = 0
        total_number = 0

        for X, Y in train_loader:
            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            # output = model.forward(X)
            hypothesis = model(X)
            cost = criterion(hypothesis, Y)
            cost.backward()
            optimizer.step()
            average_cost += cost
            '''
            predict = tc.argmax(output.data, 1)
            total_number += Y.size(0)
            train_accuracy += (predict == Y).sum().item()
            '''

        average_cost /= total_batch
        # train_accuracy /= total_number
        # valid_accuracy = evaluate_acc(valid_loader, model)
        print('[Epoch:{}] cost = {}'.format(i + 1, average_cost))
        # print('[Epoch:{}] cost = {} train_acc = {} valid_acc = {}'.format(i + 1, average_cost, train_accuracy, valid_accuracy))
        schedule.step()

    print('Training Finished.')


def test(model, test_dat):
    # model.eval()

    accuracy = 0
    with tc.no_grad():
        iter = 0
        for i in range(len(test_dat)):
            iter += 1
            X = test_dat[i][0].unsqueeze(0).to(device)
            Y = test_dat[i][1]
            prediction = model(X)
            prediction = tc.argmax(prediction, 1).item()
            if prediction == Y:
                accuracy += 1
        accuracy /= iter

    print('Accuracy:', accuracy)

    '''
    r = random.randint(0, len(test_dat) - 1)
    X_single_data = test_dat.imgs data[r:r + 1].view(-1, 3, 128, 128).float().to(device)
    Y_single_data = test_dat.targets[r:r + 1].to(device)

    print('Here shows one random example:')
    print('Label: ', Y_single_data.item())
    single_prediction = model(X_single_data)
    print('Prediction: ', tc.argmax(single_prediction, 1).item())

    plt.imshow(X_single_data[0].to('cpu').numpy())
    print('Close the window to advance.')
    plt.show()
    '''


def main():
    model = VGG(num_classes=11, init_weight=True).to(device)
    # model = ResNet().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=l2_lambda)
    schedule = scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1)

    # train(model, train_data, valid_data, criterion, optimizer, schedule, train_epoch)
    # tc.save(model.state_dict(), './mnist_model_VGG.pth')
    model.load_state_dict(tc.load('./mnist_model_VGG.pth', weights_only=True))

    '''
    print(len(train_data))
    for X, Y in valid_data:
        print(X.size())
        print(Y)
        plt.imshow(X[0].permute(1, 2, 0).to('cpu').numpy())
        tag = [key for key, value in class_to_index.items() if value == Y[0]]
        print(tag)
        plt.title(Y[0].item())
        plt.show()

        X = X.to(device)
        prediction = model(X)
        print(prediction.shape)
        prediction = tc.argmax(prediction, 1)
        print(prediction)
        break
    '''

    print(len(train_dataset))
    print(evaluate_acc(valid_data, model))
    # print(valid_dataset)
    '''
    print(len(valid_dataset))
    plt.imshow(valid_dataset[0][0].permute(1, 2, 0).to('cpu').numpy())
    tag = [key for key, value in class_to_index.items() if value == valid_dataset[0][1]]
    print(tag)
    print(valid_dataset.imgs[0])
    plt.title(valid_dataset[0][1])
    plt.show()
    '''


main()
