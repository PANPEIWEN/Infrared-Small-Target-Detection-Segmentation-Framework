# @Time    : 2022/5/31 17:19
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : scheduler.py
# @Software: PyCharm

def PolyLR(optimizer, epoch, num_epochs, base_lr, power):
    lr = base_lr * (1 - epoch / num_epochs) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
