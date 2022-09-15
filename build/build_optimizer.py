# @Time    : 2022/9/14 20:17
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : build_optimizer.py
# @Software: PyCharm
import torch.optim as optim


def build_optimizer(optimizer_name, model, lr):
    if optimizer_name == 'Adagrad':
        optimizer = optim.Adagrad(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        # optimizer = optim.Adagrad(model.parameters(), lr=0.05, weight_decay=1e-4)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        assert 'Optimizer: ' + optimizer_name + ' is not defined'
    return optimizer
