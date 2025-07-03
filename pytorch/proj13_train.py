import torch 
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

# 检查 MPS 是否可用
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ),(0.3081, ))
])
# python读取图像是pillow形式，需要先变成张量
# 同时神经网络喜欢（0，1）正态分布的量，所以需要标准化
# 再把 28*28 的原始图像变成 1*28*28 的单通道灰度图像

# 数据需要分为训练集和测试集
train_dataset = datasets.MNIST(root='../dataset/mnist',
                               train=True,
                               download=True,
                               transform=transform)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)
test_dataset = datasets.MNIST(root='../dataset/mnist',
                              train=False,
                              download=True,
                              transform=transform)
test_loader = DataLoader(dataset=test_dataset,
                         shuffle=False,
                         batch_size=batch_size)

# 批量大小为N，则输入张量的维度为 N*1*28*28
# 但是模型的输入需要变成二维矩阵，其中列数即是维度，行数是样本数

class Net(torch.nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10) # 最后是十分类问题，降到10列即是10个特征

    def forward(self, x):
        x = x.view(-1, 784) # 需要把张量变成矩阵，特征数是784，-1代表自动计算
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = self.l5(x)  # 最后一层不做激活，交叉熵损失已经做过激活
        return x 

# 利用 MPS 加速，发挥 Apple Silicon 芯片的图形处理能力
model = Net().to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(params=model.parameters(), lr=0.01, momentum=0.5)

# 把一轮训练封装成为函数
def train(epoch):
    running_loss = 0 # 存放当前的损失值
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device) # 将数据移到 MPS 上
        optimizer.zero_grad()

        y_pred = model(inputs)
        loss = criterion(y_pred, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.6f' % (epoch+1, batch_idx+1, running_loss/300))
            running_loss = 0
        
def test():
    correct = 0
    total = 0
    with torch.no_grad(): # 代码不会进行反向传播计算梯度
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device) # 将数据移到 MPS 上
            outputs = model(images)
            _, predictions = torch.max(outputs.data, dim=1) # 按照列找最大的值，返回最大值和下标（即属于第几个类）
            total += labels.size(0) # 每次从 test_loader 中取出一个批次的数据时，labels.size(0) 返回该批次的标签数量
            correct += (labels == predictions).sum().item()
    print('Accuracy on test set: %d %%' % (100 * correct / total))

def save_model():
    torch.save(model.state_dict(),f='mnist_model.pth')
    print("Model saves as mnist_model.pth")

# 训练
if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        if epoch % 2 == 0:
            test() # 每10轮训练，测试一次
        # if device.type == 'mps': # 手动清理缓存
        #     torch.mps.empty_cache()
    
    save_model()









        
