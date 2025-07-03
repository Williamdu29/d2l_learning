import numpy as np
import torch
import torch.backends

# 定义的模型与训练时的一致
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

# 加载模型参数
model.load_state_dict(torch.load('model.pth'))

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model.to(device)

# 设置模型为评估模式（不进行反向传播）
model.eval()

# 加载测试数据集
test_data = np.loadtxt(fname='diabetes_test_set_for_prediction.csv',delimiter=',',dtype=np.float32)

# 转化为Tensor
x_test = torch.from_numpy(test_data[:, :-1]).to(device)
y_test = torch.from_numpy(test_data[:,[-1]]).to(device)

# 禁止梯度计算，预测
with torch.no_grad():
    y_pred = model(x_test)

# 转化为二分类问题
y_pred_class = (y_pred>=0.5).float()

print("******预测值与真实值的对比******")
for pred,true in zip(y_pred_class[:100],y_test[:100]):
    print("预测值：",pred.item(),"真实值：",true.item())







