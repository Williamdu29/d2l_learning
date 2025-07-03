# 梯度下降算法
import numpy as np

x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]

w=1.0
#随机初始化参数，通过训练进行修改

def forward(x):
    return x*w

def cost(xs,ys):
    cost=0
    for x,y in zip(xs,ys):
        y_pred=forward(x)
        cost+=(y_pred-y)**2
    return cost/len(xs)
#返回平均损失的函数

def gradient(xs,ys):
    grad=0
    for x,y in zip(xs,ys):
        grad+=2*x*(x*w-y)
        #推导出来的求  partial cost/ partial w 的公式
    return grad/len(xs)

print("Predict brfore training:",4,forward(4))
#预测一下初始化的结果
print("--------")

learning_rate=0.01

for epoch in range(100):
    #设置迭代轮数为100
    cost_val=cost(x_data,y_data)
    grad_val=gradient(x_data,y_data)
    w-=learning_rate*grad_val
    print("Epoch:",epoch,"w=",w,"loss=",cost_val)
#经历过100轮的训练后，w的值已经向着局部最优解靠近

print("--------")
print("Predict after taining",4,forward(4))