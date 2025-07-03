import torch
import torch.nn.functional as F

x_data=torch.Tensor([[1.0],[2.0],[3.0]])
y_data=torch.Tensor([[0],[0],[1]])
#真实值改为0-1分类

class LosgisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LosgisticRegressionModel,self).__init__()
        self.linear=torch.nn.Linear(1,1)
        #因为sigmoid函数没有参数，因此不需要新传入参数，也不需要对其进行训练
        
    def forward(self,x):
        y_pred=F.sigmoid(self.linear(x))
        return y_pred
    #sigmoid计算单元：就是在线性计算单元后面添加了logistic function
   

model=LosgisticRegressionModel()

criterion=torch.nn.BCELoss(size_average=False) 
#损失函数选取交叉熵损失

optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
#选择优化器，其中model.parameters()是给参数求梯度，便于反馈时更新参数

for epoch in range(1000):
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    print(epoch,loss.item())
    #因为loss是torch.nn.MSELoss的实例化，继承的nn模块，因此调用就会产生计算图，为了避免则需要.item()
    
    optimizer.zero_grad()
    #先梯度清零再反向传播
    loss.backward()
    optimizer.step()
    #更新参数
print('--------')
print('w=',model.linear.weight.item())
print('b=',model.linear.bias.item())
print('--------')

#对于训练好的模型再做一次测试，此时的各个参数应该都是最优
x_test=torch.Tensor([4.0])
y_test=model(x_test)
print("y_pred=",y_test.data)