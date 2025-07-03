# 接proj3.py，梯度下降改为随机梯度下降，实际是对单个样本求梯度和loss
import matplotlib.pyplot as plt

x_data = [1,2,3]
y_data = [2,4,6]

w = 1

def forward(x):
    y_pred = x*w
    return y_pred

def loss(x,y):
    y_pred = forward(x)
    return (y_pred-y)*(y_pred-y)

def gradient(x,y):
    y_pred = forward(x)
    return 2*(y_pred-y)*x

print("predict before training:",4,forward(4))

lr = 0.01

epoch_list=[]
loss_list=[]

for epoch in range(100):
    for x,y in zip(x_data,y_data):
        l =loss(x,y)
        w -= lr*gradient(x,y)
    epoch_list.append(epoch)
    loss_list.append(l)
    print('epoch:',epoch, 'loss:',l, 'w:',w)

print("predict after training:",4,forward(4))

plt.plot(epoch_list,loss_list)
plt.xlabel('EPOCHES')
plt.ylabel("LOSS")
plt.show()