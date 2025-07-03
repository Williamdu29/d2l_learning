# 演示卷积运算

import torch
import torch.nn.functional as F

'''
in_channels = 5
out_channels = 10
width, height = 100, 100
batch_size = 1
kernel_size = 3

input = torch.randn(batch_size,in_channels,width,height)
# 随机生成输入图像，图像按照batch输入，所以为4维：b*c*w*h

conv_layer = torch.nn.Conv2d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size)

output = conv_layer(input)

print(input.shape)
print(output.shape)
print(conv_layer.weight.shape)
'''

'''
input = [3,4,6,5,7,
         2,4,6,8,2,
         1,6,7,8,4,
         9,7,4,6,2,
         3,7,5,4,1]

input = torch.Tensor(input).view(1,1,5,5) # 变成torch需要的输出格式

conv_layer = torch.nn.Conv2d(1,1,3,padding=1,bias=False)

kernel = torch.Tensor([1,2,3,4,5,6,7,8,9]).view(1,1,3,3)
conv_layer.weight.data = kernel.data # 初始化卷积层的参数

output = conv_layer(input)
print(output.data)
'''



class Net(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Net,self).__init__(*args, **kwargs)
        self.conv1 = torch.nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10,20,kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2) # 2*2 池化，即是把图像的大小减半
        self.fc = torch.nn.Linear(320,10)

    def forward(self,x):
        batch_size = x.size(0) # 拿出批量大小
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x.view(batch_size,-1)   # 输入全链接层的大小 batch*320    -1表示自动计算
        x = self.fc(x)
        return x
    
model = Net()

