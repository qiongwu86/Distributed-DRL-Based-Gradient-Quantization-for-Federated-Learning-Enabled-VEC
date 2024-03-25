import scipy.stats as stats
import os
import numpy as np
import tensorflow as tf
import ipdb as pdb
import matplotlib.pyplot as plt
from options import args_parser
import torch.nn as nn
import torch

args = args_parser()
LOSS_FUNC = nn.CrossEntropyLoss()

# 进行AR信道建模
def complexGaussian(row=1, col=1, amp=1.0):
    real = np.random.normal(size=[row, col])[0] * np.sqrt(0.5)
    # np.random.normal(size=[row,col])生成数据2维 第一维度包含row个数据 每个数据中又包含col个数据
    # np.sqrt(A)求A的开方
    img = np.random.normal(size=[row, col])[0] * np.sqrt(0.5)
    return amp * (real + 1j * img)  # amp相当于系数 后面计算时替换成了根号下1-rou平方    (real + 1j*img)即为误差向量e(t)


class ARModel(object):
    """docstring for AR channel Model"""

    def __init__(self, n_t=1, n_r=1, seed=123):
        self.n_t = n_t
        self.n_r = n_r
        np.random.seed([seed])
        self.H1 = complexGaussian(self.n_t, self.n_r)  # self.H就是hi 即信道增益。初始化定义.

    def sampleCh(self, dis, rho):
        for i in range(args.num_users):
            # self.H1[i] = rho[i] * self.H1[i] + complexGaussian(self.n_t, self.n_r, np.sqrt(1 - rho[i] * rho[i]))  # 这是信道更新的方式
            self.H1[i] = rho[i] * self.H1[i] + complexGaussian(1, 1, np.sqrt(1 - rho[i] * rho[i])) #因为这里是一个一个算的 所以是一个一个复高斯数生成的 所以可以直接写成1,1
        return self.H1

def one_iter(model, device, loss_func, optimizer, quantizer, train_data, num_users, epoch):
    assert num_users == len(train_data)
    model.train()
    user_gradients = [list() for _ in model.parameters()] # 4层梯度
    all_losses = []
    for user_id in range(num_users): 
        optimizer.zero_grad()
        _data, _target = train_data[user_id]
        data, target = _data.to(device), _target.to(device)
        pred = model(data)
        loss = loss_func(pred, target)
        # print(loss)
        all_losses.append(loss)
        loss.backward()
        #parameter = list(model.parameters())
        quantizer.record(user_id, epoch=epoch)
    quantizer.apply()
    # parameter = list(model.parameters())
    optimizer.step()
    #parameter1 = list(model.parameters())
    return torch.stack(all_losses).mean()

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad(): # 在该模块下,所有计算得出的tensor的requires_grad都自动设置为False
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += LOSS_FUNC(output, target).sum().item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    #    test_loss, correct, len(test_loader.dataset),
    #    100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)