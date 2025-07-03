import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]
# 人工数据集

w = 1 # initial guess

def forward(x):
    return x * w

def cost(x_val,y_val):
    loss = 0
    for x,y in zip(x_val,y_val):
        y_pred = forward(x)
        loss += (y_pred-y)*(y_pred-y)
    return loss/len(x_val)

def gradient(x_val,y_val):
    grad = 0
    for x,y in zip(x_val,y_val):
        y_pred = forward(x)
        grad += 2*x*(y_pred-y)
    return grad/len(x_val)


print("predict before training:",4,forward(4))

lr = 0.01

epoch_list=[]
loss_list=[]

for epoch in range(100):
    loss = cost(x_data,y_data)
    w -= lr*gradient(x_data,y_data)
    print('epoch:',epoch,'LOSS:',loss,'w:',w)
    loss_list.append(loss)
    epoch_list.append(epoch)


print("predict after training 100 epoches:",4,forward(4))

plt.plot(epoch_list,loss_list)
plt.xlabel("epoches")
plt.ylabel("LOSS")
plt.show()

