import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden=nn.Linear(20,256) #隐层
        self.out=nn.Linear(256,10) #输出

    def forward(self,x):
        return self.out(F.relu(self.hidden(x)))
    

X=torch.rand(2,20) #输入
print(X)
print(" ---经过多层感知机--- ")
net=MLP()
print(net(X)) #反向传播和参数初始化是自动的 

# print(torch.cuda.device_count())






