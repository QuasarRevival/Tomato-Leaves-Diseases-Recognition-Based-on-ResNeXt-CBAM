import torch as tc
import torch.optim as optim
import torch.nn as nn
import csv

import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as scheduler
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import cohen_kappa_score

from CNN_models.model_VGG import VGG
from CNN_models.model_ResNet import ResNet
from CNN_models.model_ResNeXt import ResNeXt

from test_model import predict_from_extended, predict_image, open_image

# GPU是否可用，设置随机数种子使得训练过程可复现
device = 'cuda' if tc.cuda.is_available() else 'cpu'
tc.manual_seed(415)
if device == 'cuda':
    tc.cuda.manual_seed_all(415)

# 训练集与验证集路径
train_path = 'datasets/TomatoLeavesDataset/train'
valid_path = 'datasets/TomatoLeavesDataset/valid'
test_path = 'datasets/PlantVillageTomatoLeavesDataset/val'

yolo_model_path = 'runs/detect/train2/weights/best.pt'
single_image_path = 'datasets/ExtendedTestImages/Target_spot/Ts2.jpg'

# 每次训练前都要修改记录存放路径！
root_path = "training_records/ResNeXt/after_hyperparam_optim/"

# 定义种类字典
class_to_index = {
    'Bacterial_spot': 0,
    'Early_blight': 1,
    'Healthy': 2,
    'Late_blight': 3,
    'Leaf_mold': 4,
    'Powdery_mildew': 5,
    'Septoria_leaf_spot': 6,
    'Spider_mites Two-spotted_spider_mite': 7,
    'Target_spot': 8,
    'Tomato_mosaic_virus': 9,
    'Tomato_yellow_leaf_curl_virus': 10
}

# 设置超参数
image_size = (224, 224)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

learning_rate = 0.000788
batch_size = 16
train_epoch = 50
l2_lambda = 0.000028

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
test_dataset = datasets.ImageFolder(
    root=test_path,
    transform=transform_valid
)

# 数据可训练化
train_data = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    pin_memory=True if device == 'cuda' else False
)
valid_data = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=False,
    pin_memory=True if device == 'cuda' else False
)
test_data = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True,
    pin_memory=True if device == 'cuda' else False
)


# 校验准确率
def evaluate_acc(test_loader, model, with_kappa=False):
    model.eval()
    test_correct = 0.0
    test_total = 0.0
    l = []
    p = []
    with tc.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            inputs, labels = images.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = tc.argmax(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            if with_kappa:
                l.extend(labels.to('cpu').tolist())
                p.extend(predicted.to('cpu').tolist())
    test_accuracy = test_correct / test_total
    l = np.array(l)
    p = np.array(p)
    l.reshape(-1, 1)
    p.reshape(-1, 1)
    kappa = 0
    if with_kappa:
        kappa = cohen_kappa_score(l, p, weights='quadratic')
    return test_accuracy, kappa


# 训练模型
def train(model, train_loader, valid_loader, criterion, optimizer, schedule=None, epoch=25, show_acc=True):
    print('Training starts.')
    model.train()
    total_batch = len(train_loader)

    average_cost_list = []
    train_accuracy_list = []
    valid_accuracy_list = []

    scaler = GradScaler()

    accumulate = 4

    for i in range(epoch):
        average_cost = 0
        train_accuracy = 0
        total_number = 0

        for iter, (X, Y) in enumerate(train_loader):
            X = X.to(device)
            Y = Y.to(device)

            with autocast():
                output = model.forward(X)
                cost = criterion(output, Y)
            scaler.scale(cost).backward()

            if (iter + 1) % accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            average_cost += cost.item()
            if show_acc:
                predict = tc.argmax(output.data, 1)
                total_number += Y.size(0)
                train_accuracy += (predict == Y).sum().item()

        average_cost /= total_batch
        average_cost_list.append(average_cost)
        if show_acc:
            train_accuracy /= total_number
            train_accuracy_list.append(train_accuracy)
            valid_accuracy, _ = evaluate_acc(valid_loader, model)
            valid_accuracy_list.append(valid_accuracy)
            print('[Epoch:{}] cost = {} train_acc = {} valid_acc = {}'
                  .format(i + 1, average_cost, train_accuracy, valid_accuracy))
        else:
            print('[Epoch:{}] cost = {}'.format(i + 1, average_cost))

        if schedule is not None:
            schedule.step()

    print('Training Finished.')
    return average_cost_list, train_accuracy_list, valid_accuracy_list


def main():
    print("Choose the type of the model from VGG, ResNet and ResNeXt: ")
    switch = input()

    if switch == 'VGG':
        model = VGG(num_classes=11, init_weight=True).to(device)
    elif switch == 'ResNet':
        model = ResNet(num_classes=11, init_weight=True).to(device)
    elif switch == 'ResNeXt':
        model = ResNeXt(num_classes=11, init_weight=True, dropout_rate=0.466, cardinality=16, base_width=4,
                        SE_reduction_ratio=16, SA_kernel_size=3).to(device)
    else:
        print("Wrong Choice!")
        exit(-1)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.87, 0.99), weight_decay=l2_lambda)
    schedule = scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.1)

    # 尝试使用自适应学习率衰减
    # schedule = scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    print("Load the model from local? Y for yes, N for no: ", end='')
    choose = input()
    if choose == 'Y':

        model.load_state_dict(tc.load(root_path + 'model_param_' + switch + '.pth', weights_only=True))
        model.eval()
        '''
        accuracy, kappa = evaluate_acc(test_data, model, with_kappa=True)
        print("Accuracy on test dataset: ", accuracy)
        print("Kappa score: ", kappa)
        '''
        prediction = predict_from_extended(img_path=single_image_path, yolo_path=yolo_model_path, model=model)

        if prediction is None:
            image = open_image(image_path=single_image_path)
            prediction = predict_image(model, image)
        tag = [key for key, value in class_to_index.items() if value == prediction]
        print("Prediction: ", tag[0])

    print("Train the model? Y for yes, N for no: ", end='')
    choose = input()
    if choose == 'Y':
        acl, tal, val = train(model, train_data, valid_data, criterion, optimizer, schedule, train_epoch)
        # acl = [i for i in range(25)]
        # tal = [i for i in range(25)]
        # val = [i for i in range(25)]
        epc = [i for i in range(1, train_epoch+1)]
        test_acc, kappa = evaluate_acc(test_data, model, with_kappa=True)
        print("Accuracy on test dataset: ", test_acc)
        print("Kappa score: ", kappa)
        csv_data = zip(epc, acl, tal, val)

        with open(root_path + 'train_log.csv', "w", newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Epoch", "Cost", "Train_acc", "Valid_acc"])
            writer.writerows(csv_data)
            # writer.writerow([test_acc])

    print("Save the model? Y for yes, N for no: ", end='')
    choose = input()
    if choose == 'Y':
        tc.save(model.state_dict(), root_path + 'model_param_' + switch + '.pth')

    '''
    print(len(test_data))
    for X, Y in test_data:
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


if __name__ == '__main__':
    main()
