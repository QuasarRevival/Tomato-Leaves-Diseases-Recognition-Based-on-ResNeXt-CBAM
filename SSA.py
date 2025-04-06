import numpy as np
import torch as tc
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

from CNN_models.model_ResNeXt import ResNeXt
from main import transform_train, transform_valid, class_to_index


# 数据集路径
train_path = 'datasets/PlantVillageTomatoLeavesDataset/train'
valid_path = 'datasets/PlantVillageTomatoLeavesDataset/valid'

# 数据集加载
train_dataset = datasets.ImageFolder(train_path, transform=transform_train)
valid_dataset = datasets.ImageFolder(valid_path, transform=transform_valid)

# 设置SSA算法参数
population_size = 20  # 种群大小
dim = 10              # 参数维度
max_iter = 10         # 最大迭代次数
ST = 0.7              # 安全阈值


# 定义超参数范围，即超参数解空间
param = {
    'learning_rate': [0.01, 0.0001],       # 学习率
    'beta1': [0.99, 0.8],                  # Adam优化器的beta1参数
    'beta2': [0.9999, 0.98],               # Adam优化器的beta2参数
    'weight_decay': [0.0001, 0.000001],    # 权重衰减
    'dropout_rate': [0.6, 0.3]             # dropout率
}
ub = [0.01, 0.99, 0.9999, 0.0001, 0.6]  # 上界
lb = [0.0001, 0.8, 0.98, 0.000001, 0.3]  # 下界


# 适应度函数
def fitness_function(params):
    model = ResNeXt(dropout_rate=params[4], num_classes=11)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)

    # 快速训练验证
    for epoch in range(2):  # 减少训练轮次加速评估
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # 测试准确率
    correct = 0
    with tc.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct / len(test_loader.dataset)
    return accuracy


# SSA算法实现
def ssa_optimizer(lb, ub, population_size, dim, max_iter, ST):
    # 初始化种群
    population = np.random.uniform(low=lb, high=ub, size=(population_size, dim))
    fitness = np.zeros(population_size)

    best_params = None
    best_fitness = -np.inf

    for iter in range(max_iter):
        # 计算适应度
        for i in range(population_size):
            fitness[i] = fitness_function(population[i])

            if fitness[i] > best_fitness:
                best_fitness = fitness[i]
                best_params = population[i].copy()

        # 麻雀分类：发现者、跟随者、警戒者
        idx_sorted = np.argsort(-fitness)
        PD = int(0.2 * population_size)  # 发现者数量
        SD = int(0.1 * population_size)  # 警戒者数量

        # 发现者位置更新
        R2 = np.random.rand()
        for i in range(PD):
            if R2 < ST:
                alpha = np.random.rand()
                population[idx_sorted[i]] *= np.exp(-i / (alpha * max_iter))
            else:
                population[idx_sorted[i]] += np.random.normal(0, 1)

        # 跟随者位置更新
        for i in range(PD, population_size):
            if i > population_size / 2:
                population[idx_sorted[i]] *= \
                    np.exp((population[population_size - 1] - population[idx_sorted[i]]) / (i ** 2))
            else:
                L = np.ones(dim)
                A = L.T @ np.linalg.pinv(L @ L.T) @ L
                population[idx_sorted[i]] += (best_params - population[idx_sorted[0]]) @ A

        # 警戒者位置更新
        for i in range(SD):
            population[idx_sorted[-i - 1]] = best_params + np.random.normal(0, 1) * (
                        best_params - population[idx_sorted[-i - 1]])

        # 边界处理
        population = np.clip(population, lb, ub)

        print(f"Iteration {iter + 1}, Best Accuracy: {best_fitness:.4f}")

    return np.round(best_params).astype(int)


# 运行优化
if __name__ == "__main__":
    print("Before optimization:")

    optimized_params = ssa_optimizer(lb, ub, population_size, dim, max_iter, ST)
    print(f"Optimized CNN Parameters: {optimized_params}")
    print("Final model structure:")
    # print(CNN(optimized_params))
