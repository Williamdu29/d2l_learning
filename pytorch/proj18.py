import torch

# 检查是否有MPS加速器可用
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

num_class = 4
input_size = 4
hidden_size = 8
embedding_size = 10
num_layers = 2
seq_len = 5
batch_size = 1 

idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]  # 'hello'
y_data = [3, 1, 2, 3, 2]  # 'ohlol'

# 将输入和标签数据移动到MPS设备
inputs = torch.LongTensor(x_data).unsqueeze(0).to(device)  # 增加一个维度表示batch
labels = torch.LongTensor(y_data).to(device)

# 定义模型
class Model(torch.nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.emb = torch.nn.Embedding(input_size, embedding_size)
        self.rnn = torch.nn.RNN(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                                batch_first=True)
        self.fc = torch.nn.Linear(in_features=hidden_size, out_features=num_class)

    def forward(self, x):
        batch_size = x.size(0)
        # 正确地初始化hidden，大小为(num_layers, batch_size, hidden_size)
        hidden = torch.zeros(num_layers, batch_size, hidden_size).to(device)  
        x = self.emb(x)  # 进行嵌入
        x, _ = self.rnn(x, hidden)  # RNN前向传播
        x = self.fc(x)  # 全连接层

        return x.view(-1, num_class)  # 将输出形状调整为(batch * seq_len, num_class)
    
net = Model().to(device)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=net.parameters(), lr=0.05)

# 开始训练
for epoch in range(100):
    optimizer.zero_grad()  # 清空梯度
    outputs = net(inputs)  # 前向传播
    loss = criterion(outputs, labels)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

    _, idx = outputs.max(dim=1)  # 获取预测的索引
    idx = idx.cpu().numpy()  # 将结果从设备中取出并转换为 numpy 格式
    print('predicted:', ''.join([idx2char[x] for x in idx]), end='')
    print(', epoch[%d/100] loss=%.4f' % (epoch + 1, loss.item()))  
