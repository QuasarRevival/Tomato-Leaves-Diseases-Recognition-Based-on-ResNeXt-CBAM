import pandas as pd
import optuna
import numpy as np
import torch as tc
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from CNN_models.model_ResNeXt import ResNeXt

from main import transform_train, transform_valid, train, evaluate_acc

batch_size = 16
device = 'cuda' if tc.cuda.is_available() else 'cpu'
tc.manual_seed(415)
if device == 'cuda':
    tc.cuda.manual_seed_all(415)

# 数据集路径
path = './datasets/TomatoLeavesDataset/train'

# 数据集加载
train_dataset = datasets.ImageFolder(path, transform=transform_train)
test_dataset = datasets.ImageFolder(
    root='./datasets/TomatoLeavesDataset/valid',
    transform=transform_valid
)

# 数据集划分
train_dataset, _ = random_split(train_dataset, (500, len(train_dataset) - 500))
test_dataset, _ = random_split(test_dataset, (500, len(test_dataset) - 500))

# 数据可训练化
train_data = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    pin_memory=True if device == 'cuda' else False
)
test_data = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False,
    pin_memory=True if device == 'cuda' else False
)


class ParamOptimizer:
    def __init__(self, max_iter=20):
        self.study = optuna.create_study(direction='maximize', storage='sqlite:///optimize.db',
                                         load_if_exists=True)
        self.max_iter = max_iter
        self.best_fitness = -np.inf
        self.best_params = None

    def _evaluate_fitness(self, params):
        # 训练ResNeXt并返回验证准确率
        model = ResNeXt(
            num_classes=11,
            dropout_rate=params['dropout_rate'],
            cardinality=params['cardinality'],
            base_width=params['base'],
            SE_reduction_ratio=params['SE_reduction_ratio'],
            SA_kernel_size=params['SA_kernel_size'],
            init_weight=True
        ).to(device)
        optimizer = tc.optim.Adam(model.parameters(), lr=params['lr'],
                                  betas=(params['beta1'], params['beta2']),
                                  weight_decay=params['wd'])
        criterion = tc.nn.CrossEntropyLoss().to(device)
        # 训练代码
        train(model=model, train_loader=train_data, valid_loader=None, criterion=criterion,
              optimizer=optimizer, schedule=None, epoch=10, show_acc=False)
        # 评估代码
        val_acc, _ = evaluate_acc(test_loader=test_data, model=model, with_kappa=False)
        return val_acc

    def search(self):
        self.study.optimize(self._objective, n_trials=self.max_iter)
        self.best_params = self.study.best_params
        self.best_fitness = self.study.best_value
        return self.best_params, self.best_fitness

    def _objective(self, trial):
        params = {
            'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
            'beta1': trial.suggest_float('beta1', 0.8, 0.99),
            'beta2': trial.suggest_float('beta2', 0.98, 0.9999),
            'wd': trial.suggest_float('wd', 0.000001, 0.0001, log=True),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.3, 0.6),
            'cardinality': trial.suggest_categorical('cardinality', [16, 32, 64]),
            'base': trial.suggest_categorical('base', [2, 4, 8]),
            'SE_reduction_ratio': trial.suggest_categorical('SE_reduction_ratio', [8, 16, 32]),
            'SA_kernel_size': trial.suggest_categorical('SA_kernel_size', [3, 5, 7, 9]),
        }
        accuracy = self._evaluate_fitness(params)
        return accuracy


if __name__ == '__main__':

    print("Before optimization:")
    model = ResNeXt(num_classes=11, init_weight=True).to(device)
    optimizer = tc.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.00001)
    criterion = tc.nn.CrossEntropyLoss().to(device)
    cost_log, _, _ = train(model=model, train_loader=train_data, valid_loader=None, criterion=criterion,
                           optimizer=optimizer, schedule=None, epoch=10, show_acc=False)
    val_acc, _ = evaluate_acc(test_loader=test_data, model=model, with_kappa=False)
    print("Validation accuracy before optimization: ", val_acc)
    cost_log = {'cost_log': cost_log}
    cost_log = pd.DataFrame(cost_log)
    cost_log.to_csv('hyper_param_optim_record/before_optimization_cost_log.csv')
    val_acc = pd.DataFrame({'val_acc': [val_acc]})
    val_acc.to_csv('hyper_param_optim_record/before_optimization_val_acc.csv')

    del model
    del optimizer
    del criterion

    print("Starting hyperparameter optimization...")
    hyper_optimizer = ParamOptimizer(max_iter=250)
    best_params, best_fitness = hyper_optimizer.search()

    print("Best parameters found: ", best_params)
    print("Best accuracy: ", best_fitness)

    best_params = pd.DataFrame(best_params, index=[0])
    best_params.to_csv('hyper_param_optim_record/best_params.csv')
    val_acc = pd.DataFrame({'val_acc': [best_fitness]})
    val_acc.to_csv('hyper_param_optim_record/best_val_acc.csv')

    fig = optuna.visualization.plot_optimization_history(hyper_optimizer.study)
    fig.show()
    fig = optuna.visualization.plot_param_importances(hyper_optimizer.study)
    fig.show()

    drop_id = ['datetime_start', 'datetime_complete', 'duration', 'state']
    df = hyper_optimizer.study.trials_dataframe().drop(drop_id, axis=1)
    df.to_csv('hyper_param_optim_record/optuna_trials.csv', index=False)
