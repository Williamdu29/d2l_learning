import torch
import matplotlib.pyplot as plt


# prepare dataset
x_data = torch.Tensor([[1],[2],[3]])
y_data = torch.Tensor([[2],[4],[6]])

# design model
class LinearModel(torch.nn.Module):
    def __init__(self) -> None:
        super(LinearModel,self).__init__()
        # 定义网络的层
        self.linear = torch.nn.Linear(1,1,bias=True)

    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred
    

model = LinearModel()

# loss and optimizer
criterion = torch.nn.MSELoss(size_average=False)
opimizer = torch.optim.SGD(params=model.parameters(),lr=0.01)

epoch_list=[]
loss_list=[]

# training cycle
for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)

    loss_list.append(loss.item())

    print(epoch,loss.item())

    opimizer.zero_grad()

    loss.backward()
    opimizer.step()

    epoch_list.append(epoch)

print("--------")
print('w :',model.linear.weight.item())
print('b :',model.linear.bias.item())
print("--------")

# prediction
print("prediction after training......")
x_test = torch.Tensor([[4]])
y_test = model(x_test)
print("y_pred =",y_test.data)

plt.plot(epoch_list,loss_list)
plt.xlabel("epoches")
plt.ylabel("loss")
plt.show()





    
