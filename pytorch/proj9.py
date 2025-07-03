import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# prepare dataset
x_data = torch.Tensor([[1],[2],[3]])
y_data = torch.Tensor([[0],[0],[1]])

# define model
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self) -> None:
        super(LogisticRegressionModel,self).__init__()
        self.linear = torch.nn.Linear(1,1)
        # sigmoid函数没有参数可供训练，直接初始化即可，不必出现在initial函数

    def forward(self,x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred
    
model = LogisticRegressionModel()

# loss and optimizer
criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(params=model.parameters(),lr=0.01)

loss_list=[]
epoch_list=[]

# training cycle
# 程序化
for epoch in range(1000):
    y_pred = model(x_data) # 调用model即就是调用了前向函数
    loss = criterion(y_pred,y_data)

    loss_list.append(loss.item())

    print(epoch,loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    epoch_list.append(epoch)


plt.plot(epoch_list,loss_list)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()




# use trained model to predict

x = np.linspace(0,10,200)
x_val = torch.Tensor(x).view((200,1)) # 变成200行1列的tensor
y_val = model(x_val)
y = y_val.data.numpy() # 再把tensor转化为numpy

plt.plot(x,y)
plt.xlabel("learing hours")
plt.ylabel("possibility of passing")
plt.grid()
plt.show()
