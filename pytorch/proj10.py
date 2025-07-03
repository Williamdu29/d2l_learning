import numpy as np
import torch
import matplotlib.pyplot as plt

# 加载糖尿病数据集
xy = np.loadtxt(fname='diabetes.csv.gz',dtype=np.float32, delimiter=',')
x_data = torch.from_numpy(xy[: , :-1]) # 遍历所有的行，列数从0到-1:拿出前8列
y_data = torch.from_numpy(xy[ :, [-1]]) # 遍历所有的行，列数只拿出-1列


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super(Model,self).__init__()

        self.linear1 = torch.nn.Linear(8,6)
        self.linear2 = torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4,1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x
    

model = Model()

learning_rate = 0.01

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(params=model.parameters(),lr=learning_rate)

loss_list = []
epoch_list = []

for epoch in range(1000):
    epoch_list.append(epoch)
    # FORWARD
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    loss_list.append(loss.item())
    print(epoch,loss.item())

    # BACKWARD
    optimizer.zero_grad()
    loss.backward()

    # UPDATE
    optimizer.step()


plt.plot(epoch_list,loss_list)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()


