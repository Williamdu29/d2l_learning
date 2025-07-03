# use dataset and dataloader to load data

import torch
import numpy as np
import torch.backends
import torch.backends.mps
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class DiabetesDataset(Dataset):
    def __init__(self,filepath) -> None:
        super(DiabetesDataset,self).__init__()
        
        xy = np.loadtxt(fname=filepath,delimiter=',',dtype=np.float32)
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])
        
        self.len = xy.shape[0]
    
    # 1.支持索引
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    # 2.支持长度
    def __len__(self):
        return self.len
    
dataset = DiabetesDataset('diabetes.csv.gz')
train_loader = DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=2)

# 利用 apple silicon M2 芯片加速
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


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
    

model = Model().to(device)

learning_rate = 0.001

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(params=model.parameters(),lr=learning_rate)

if __name__ =='__main__':
    for epoch in range(300):
        for i,data in enumerate(train_loader,start=0):
            inputs,labels = data
            inputs, labels = inputs.to(device), labels.to(device) 
            # 1.FORWARD
            y_pred = model(inputs)
            loss = criterion(y_pred,labels)

            print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")

            # 2.BACKWARD
            optimizer.zero_grad()
            loss.backward()

            # 3.UPDATE
            optimizer.step()

    # 保存训练完成的模型参数
    torch.save(model.state_dict(), 'model.pth')
    print("模型已保存为 'model.pth'")




        

        


