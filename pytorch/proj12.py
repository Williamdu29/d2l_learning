# 多分类问题——softMax函数的实现

import numpy as np
import torch 

# numpy 实现
'''
y = np.array([1,0,0]) # one_hot 编码
z = np.array([0.2,0.1,-0.1])

y_pred = np.exp(z)/np.exp(z).sum()
loss = (-y * np.log(y_pred)).sum()
print(loss)
'''


# pytorch 实现
'''
y = torch.LongTensor([0]) # 在 one_hot 编码中使第0个元素为1，即是计算第0个分类的损失值
z = torch.Tensor([[0.2,0.1,-0.1]])

criterion = torch.nn.CrossEntropyLoss()
loss = criterion(z,y)
print(loss.item())
'''


# 问题：有两个预测，他们预测了三个样本，分别属于第2类，第0类，第1类的预测值，比较他们之间的损失值

Y_pred1 = torch.Tensor([[0.1,0.2,0.9],     # 第一个预测中，0.9最大，即是认为是第2类 
                        [1.1,0.1,0.2],     # 认为是第0类
                        [0.2,2.1,0.2]])    # 认为是第1类
Y_pred2 = torch.Tensor([[0.8,0.2,0.3],
                        [0.2,0.3,0.5],
                        [0.2,0.2,0.5]])
Y = torch.LongTensor([2,0,1]) # one_hot编码中把预测的类置为1
criterion = torch.nn.CrossEntropyLoss()

loss1 = criterion(Y_pred1,Y)
loss2 = criterion(Y_pred2,Y)

print("LOSS1=",loss1.item(),"\nLOSS2=",loss2.item())