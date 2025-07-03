# processing minst dataset

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

# perpare minst dataset
train_set = torchvision.datasets.MNIST(root='../data/minst',train=True,download=True)
test_set = torchvision.datasets.MNIST(root='../date/minst',train=False,download=True)


# 定义sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 创建x值范围
x = np.linspace(-10, 10, 100)

# 计算sigmoid值
y = sigmoid(x)

# 绘制sigmoid函数
plt.figure(figsize=(8, 5))
plt.plot(x, y, label='Sigmoid Function')
plt.title('Sigmoid Function')
plt.xlabel('x')
plt.ylabel('Sigmoid(x)')
plt.grid(True)
plt.legend()
plt.show()


# sigmoid函数没有参数可供训练，直接初始化即可，不必出现在initial函数



