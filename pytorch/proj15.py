# google net 的实现

import torch
import torch.nn.functional as F
import torch.nn as nn

class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA,self).__init__()

        self.branch1x1 = nn.Conv2d(in_channels,16,kernel_size=1)

        self.branch5x5_1 = nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(in_channels,24,kernel_size=5,padding=2)

        self.branch3x3_1 = nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16,24,kernel_size=3,padding=1)
        self.branch3x3_3 = nn.Conv2d(24,24,kernel_size=3,padding=1)

        self.branch_pool = nn.Conv2d(in_channels,24,kernel_size=1)


    def forward(self,x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x,kernel_size=3,stride=1,padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        outputs = torch.cat(outputs,dim=1) # 沿着channel维度拼接

        return outputs
    
class Net1(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Net1, self).__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = nn.Conv2d(88,20,kernel_size=5)

        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)

        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(1408,10)

    def forward(self,x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        # 输出为88维
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        # 输出为88维
        x = x.view(in_size,-1)
        # 展为向量送进全链接层
        x = self.fc(x)
        
        return x 
    



# residual net 的实现

class ResidualBlock(nn.Module):
    def __init__(self, channels) -> None:
        super(ResidualBlock, self).__init__()
        self.channels = channels

        self.conv1 = nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(channels,channels,kernel_size=3,padding=1)

    def forward(self,x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(y + x) # 先求和再激活
     

class Net2(nn.Module):
    def __init__(self) -> None:
        super(Net2,self).__init__()
        self.conv1 = nn.Conv2d(1,16,kernel_size=5)
        self.conv2 = nn.Conv2d(16,32,kernel_size=5)
        self.mp = nn.MaxPool2d(2)

        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)

        self.fc = nn.Linear(512,10) # 10分类问题

    def forward(self,x):
        in_size = x.size(0)
        x = self.ap(F.relu(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.rblock2(x)
        x = x.view(in_size,-1)
        x = self.fc(x)

        return x
    
    