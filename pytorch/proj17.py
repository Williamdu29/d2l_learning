# seq2seq 
# 'hello' -> 'ohlol'

import torch 

# 检查是否有MPS加速器可用
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

batch_size = 1
hidden_size = 4
input_size = 4

# 字符列表
idx2char = ['e','h','l','o']

x_data = [1,0,2,2,3] # 'hello'
y_data = [3,1,2,3,2] # 'ohlol'

one_hot_lookup = [[1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0],
                  [0,0,0,1]]
# x_data 中数字是几，就把独热查找向量中的第几行拿走，对应为1的index就是代表的字符
x_onehot = [one_hot_lookup[x] for x in x_data]
# print(x_onehot)

inputs = torch.Tensor(x_onehot).view(-1,batch_size,input_size).to(device)
labels = torch.LongTensor(y_data).view(-1,1).to(device)  # label变为（seq，1）

# 定义模型
class Model(torch.nn.Module):
    def __init__(self, batch_size, input_size, hidden_size) -> None:
        super(Model,self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnncell = torch.nn.RNNCell(input_size=self.input_size,hidden_size=self.hidden_size)

    def forward(self,input,hidden):
        hidden = self.rnncell(input,hidden)
        return hidden
    
    def init_hidden(self):
        return torch.zeros(self.batch_size,self.hidden_size).to(device)
    
net = Model(batch_size,input_size,hidden_size).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=net.parameters(),lr=0.01)

for epoch in range(100):
    loss = 0
    optimizer.zero_grad()
    hidden = net.init_hidden()
    # 初始化h0
    print('predicted string: ',end=',')
    for input, label in zip(inputs,labels): # 由inputs和label的形状知道遍历的是seq这个维度
        hidden = net(input,hidden)
        loss  += criterion(hidden,label)

        _, idx = hidden.max(dim=1)
        print(idx2char[idx.item()],end='')
    loss.backward()
    optimizer.step()
    print(', epoch[%d/100] loss=%.4f' %((epoch+1),loss.item()))  
    

        
