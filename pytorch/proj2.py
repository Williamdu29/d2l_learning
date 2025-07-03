import torch
import matplotlib.pyplot as plt
import numpy as np

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]


def forward(x):
    return x * w

def loss(x,y):
    y_pred = forward(x)
    loss = (y-y_pred)*(y-y_pred)
    return loss

w_list = []
mse_list = []

for w in np.arange(0.0,4.1,0.1): #循环便利每一个w
    print("w=",w)
    loss_sum = 0
    for x,y in zip(x_data,y_data): #在每一个给定的w下，再遍历人工数据集x，y
        loss_sum += loss(x,y) #先不做均值
    print('losss =',loss_sum/3)
    w_list.append(w)
    mse_list.append(loss_sum/3)

plt.plot(w_list,mse_list)
plt.xlabel('w')
plt.ylabel('loss')
plt.show()

    










