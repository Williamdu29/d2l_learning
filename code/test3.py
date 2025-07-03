import torch
from torch import nn
import matplotlib.pyplot as plt
from d2l import torch as d2l

T = 1000
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))

# 使用matplotlib进行绘图
# plt.figure(figsize=(6, 3))
# plt.plot(time.numpy(), x.numpy())
# plt.xlabel('time')
# plt.ylabel('x')
# plt.xlim([1, 1000])
# plt.show()  # 确保调用plt.show()来显示图形

#马尔可夫假设

#接下来，我们将这个序列转换为模型的特征－标签（feature-label）对。 基于嵌入维度，我们将数据映射为数据对y_t=x_t,其中x_t=[x_(t-tau),...,x_(t-1)]
#这比我们提供的数据样本少了tau个， 因为我们没有足够的历史记录来描述前个数据样本。 
#一个简单的解决办法是：如果拥有足够长的序列就丢弃这几项； 
#另一个方法是用零填充序列。 在这里，我们仅使用前600个“特征－标签”对进行训练。
tau=4
features=torch.zeros((T-tau,tau)) #0张量，代表样本数和特征数
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = x[tau:].reshape((-1, 1))

batch_size, n_train = 16, 600
# 只有前n_train个样本用于训练
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)


#利用MLP作为网络架构

# 初始化网络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# 一个简单的多层感知机
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),#输入的特征为4，做一个10层的隐层
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net

# 平方损失。注意：MSELoss计算平方误差时不带系数1/2
loss = nn.MSELoss(reduction='none')

#训练
def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)#Adam分类器
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)

#预测（单步预测）
onestep_preds = net(features)

plt.figure(figsize=(6, 3))
plt.plot(time.numpy(), x.detach().numpy(), label='data')
plt.plot(time[tau:].numpy(), onestep_preds.detach().numpy(), label='1-step preds')
plt.xlabel('time')
plt.ylabel('x')
plt.xlim([1, 1000])
plt.legend()
plt.show()

#无法进行长期数据预测