

import torch 
'''
# using RNNCell to make a RNN cycle

batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2

cell = torch.nn.RNNCell(input_size=input_size,hidden_size=hidden_size)

# (seq, batch_size, input_size)
dataset = torch.randn(seq_len,batch_size,input_size) #这行代码创建了一个形状为 (seq_len, batch_size, input_size) 的随机张量，用于模拟批量序列数据输入

# 初始化隐藏层 h_0 的维度
hidden = torch.zeros(batch_size,hidden_size)

for idx, input in enumerate(dataset): #遍历第0个维度即seq
    print('='*20,idx,'='*20) 

    print("InputSize:",input.shape)

    hidden = cell(input,hidden)
    
    print("OutputSize:",hidden.shape)
    print(hidden)
'''


'''
# using RNN to make the RNN cycle

batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2
num_layers = 1

cell = torch.nn.RNN(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)

# (seq, batchsize, inputsize)
inputs = torch.randn(seq_len,batch_size,input_size)
hidden = torch.zeros(num_layers,batch_size,hidden_size)

out, hidden = cell(inputs,hidden)

print('OutputSize:',out.shape)
print('Output:',out)
print('HiddenSize:',hidden.shape)
print('Hidden:',hidden)
'''



