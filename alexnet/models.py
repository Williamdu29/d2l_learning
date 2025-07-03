# contains total 8 layers with 5 conv and 3 dense
# input images size = 256 x 256
# softmax after last layer
# Response Normalization layers applied only after 1st and 2nd convs 
# follow MaxPooling layer after RN and the 5th conv
# apply ReLU after every MaxPooling and the fully-connected
# 
# --CNN layers--
# 1st: input_dim = 224 x 224 x 3, output_dim = 11 x 11 x 3, kernels = 96, stride= 4
# 2ed: input_dim = 11 x 11 x3, output_dim = 5 x 5 x 48, kernels = 256, stride = 1
# 3rd: input_dim = 5 x 5 x 48, output_dim = 3 x 3 x256, kernels = 348, stride = 1
# 4th: input_dim = 3 x 3 256, output_dim = 3 x 3 x192, kernels = 348, stride = 1
# 5th: input_dim = 3 x 3 x 192, output_dim= 3 x 3 x 192, kernels=256, stride =1
 
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#define model parameters
NUM_EPOCHS = 90 #original papaer
BATCH_SIZE = 128 
MOMENTUM = 0.9
LR_DECAY = 0.005
LR_INIT = 0.01
IMAGE_DIM = 227
NUM_CLASSES = 1000
DEVICES_IDS = [0,1,2,3] #this is the code for gpus to use

# modify this to point to your data directory
INPUT_ROOT_DIR = 'alexnet_data_in'
TRAIN_IMG_DIR = 'alexnet_data_in/imagenet'
OUTPUT_DIR = 'alexnet_data_out'
LOG_DIR = OUTPUT_DIR + '/tblogs'  # tensorboard logs
CHECKPOINT_DIR = OUTPUT_DIR + '/models'  # model checkpoints

# make checkpiont path directory
os.makedirs(CHECKPOINT_DIR,exist_ok=True)
'''
name (必需): 要创建的目录路径。
mode (可选): 设置目录的权限。默认为 0o777，实际的模式是 0o777 减去当前 umask。
exist_ok (可选): 如果 exist_ok 为 True，则如果目标目录已存在，函数不会引发 FileExistsError 异常。默认为 False，即如果目标目录已存在，会引发异常。
'''






class AlexNet(nn.Module):
    def __init__(self,num_classes=1000) -> None:
        super(AlexNet,self).__init__()
        # input size should be : (b x 3 x 227 x 227)
        # The image in the original paper states that width and height are 224 pixels, but the dimensions after first convolution layer do not lead to 55 x 55.  
        self.net = nn. Sequential(
            nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11, stride=4), # b x 96 x 55 x 55
            nn.ReLU(),
            nn.LocalResponseNorm(size=5,alpha=0.001,beta=0.75,k=2),
            nn.MaxPool2d(kernel_size=3,stride=2),# b x 96 x 27 x 27
            nn.Conv2d(in_channels=96,out_channels=256, kernel_size=5,padding=2), # b x 96 x 27 x 27
            nn.ReLU(),
            nn.LocalResponseNorm(size=5,alpha=0.001,beta=0.75,k=2),
            nn.MaxPool2d(kernel_size=3,stride=2), # b x 256 x 13 x 13
            nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,padding=1), # b x 384 x 13 x 13
            nn.ReLU(),
            #The third, fourth, and fifth convolutional layers are connected to one another without any intervening pooling or normalization layers
            nn.Conv2d(384,384,3,padding=1), # b x 384 x 13 x 13
            nn.ReLU(),
            nn.Conv2d(384,256,3,padding=1), # b x 256 x 13 x 13
            nn.MaxPool2d(kernel_size=3,stride=2), # b x 256 x 6 x 6
        )

        # classifier is just a name for linear layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5,inplace=True),
            nn.Linear(in_features=(256*6*6),out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5,inplace=True),
            nn.Linear(in_features=4096,out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096,out_features=num_classes),
        )

        self.init_bias()


        def init_bias(self):
            for layer in self.net():
                if isinstance(layer,nn.Conv2d): #只有卷积层需要初始化权重和偏置
                    nn.init.normal_(layer.weight,mean=0,std=0.01)
                    nn.init.constant_(layer.bias,0)
                # We initialized the neuron biases in the second, fourth, and fifth convolutional layers, as well as in the fully-connected hidden layers, with the constant 1
                nn.init.constant_(self.net[4].bias,1)
                nn.init.constant_(self.net[10].bias,1)
                nn.init.constant_(self.net[12].bias,1)
                #索引从0开始

        def forward(self,x):
            """
        Pass the input through the net.

        Args:
            x (Tensor): input tensor

        Returns:
            output (Tensor): output tensor
        """
            x=self.net(x)
            # reduce the dimensions for linear layer input
            x=x.view(-1,256*6*6) # -1使得这个维度自动计算以匹配原始数据的元素总数和其它指定的维度大小

            return self.classifier(x)
        
if __name__ == '__main__':
    # print the seed value
    seed = torch.initial_seed()
    print('Used seed : {}'.format(seed))
    #torch.initial_seed() 函数用于获取 PyTorch 当前使用的随机种子。这个种子是 PyTorch 随机数生成器的初始种子，它控制着所有随机操作的基础，如参数初始化、数据打乱等，从而保证实验的可重复性。

    tbwriter = SummaryWriter(log_dir=LOG_DIR)
    print('TensorboardX summary writer created')

    #creat model
    alexnet = AlexNet(num_classes=NUM_CLASSES).to(device)

    #train on multipul GPUs
    alexnet = torch.nn.parallel.DataParallel(alexnet,device_ids=DEVICES_IDS)
    print(alexnet)
    print('AlexNet created')


    #create dataset and dataloader
    dataset = datasets.ImageFolder(TRAIN_IMG_DIR, transforms.Compose([
        # transforms.RandomResizedCrop(IMAGE_DIM, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.CenterCrop(IMAGE_DIM),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    print('Dataset created')

    dataloader = data.DataLoader(
        dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=BATCH_SIZE)
    print('Dataloader created')

     #create optimizer 
    optimizer = optim.Adam(params=alexnet.parameters(),lr=0.0001)
    print("Oprimizer created")
    
    # multiply LR by 1/10 after 30 epochs
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30,gamma=0.1)
    print("LR Scheduler created")

    #now we can start training!
    print("Start training...")
    total_steps = 1
    # 初始化计数器 total_steps=1，记录训练的步数
    for epoch in range(NUM_EPOCHS):
        lr_scheduler.step()
        # 在每个epoch开始时更新学习率调度器。这行代码调整优化器的学习率，通常依据预定义的策略（如逐步衰减)

        for imgs, classes in dataloader:
            imgs, classes = imgs.to(device),classes.to(device)

            # clculate the loss
            output = alexnet(imgs)
            # 前向传播得到的分类结果，其形状为 [batch_size, num_classes]

            loss = F.cross_entropy(output,classes)
            #交叉熵损失，适用于多分类问题

            # update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() # 更新模型参数

            # log the information and add to tensorboard
            if total_steps % 10 == 0:
                with torch.no_grad():
                    _, preds = torch.max(output, 1) # 从模型中提取预测的类别 
                    '''
                    该函数返回两个对象：
                    1. 最大值的张量（在这个调用中被忽略，使用 _ 来接收，表示我们对此不感兴趣）。
                    2. 索引的张量 preds，包含每个样本得分最高的类别的索引。
                    '''

                    accuracy = torch.sum(preds == classes) # 预测正确的样本数
 
                    print('Epoch: {} \tStep: {} \tLoss: {:.4f} \tAcc: {}'
                        .format(epoch + 1, total_steps, loss.item(), accuracy.item()))
                    tbwriter.add_scalar('loss', loss.item(), total_steps)
                    tbwriter.add_scalar('accuracy', accuracy.item(), total_steps)


            # print out gradient values and parameter average values
            if total_steps % 100 == 0:
                with torch.no_grad():
                    # print and save the grad and parameters
                    # also print and save paramater values
                    print("*" * 10)
                    for name , parameter in alexnet.named_parameters():
                        #named_parameters() 方法返回一个生成器，包含参数的名字和参数本身
                        if parameter.grad is not None:
                            avg_grad = torch.mean(parameter.grad)
                            print('\t{} - grad_avg: {}'.format(name, avg_grad))
                            #打印参数名和对应的梯度
                            tbwriter.add_scalar('grad_avg/{}'.format(name), avg_grad.item(), total_steps)
                            tbwriter.add_histogram('grad/{}'.format(name),
                                    parameter.grad.cpu().numpy(), total_steps)
                            
                        if parameter.data is not None:
                            avg_weight = torch.mean(parameter.data)
                            print('\t{} - param_avg: {}'.format(name, avg_weight))
                            tbwriter.add_histogram('weight/{}'.format(name),
                                    parameter.data.cpu().numpy(), total_steps)
                            tbwriter.add_scalar('weight_avg/{}'.format(name), avg_weight.item(), total_steps)

                        
                        total_steps+=1

                    
                    # save checkpoints
                    '''
                    Checkpointing 是一种重要的技术，
                    用于在训练过程中定期保存模型的状态，以便可以在训练中断时恢复，
                    或者用于后续的模型评估和比较
                    '''
                    checkpoint_path = os.path.join(CHECKPOINT_DIR,'alexnet_states_e{}.pkl'.format(epoch+1))
                    # CHECKPOINT_DIR 是一个字符串，包含了用于存储checkpoint文件的目录的路径
                    # 'alexnet_states_e{}.pkl'.format(epoch+1) 创建一个字符串，表示checkpoint文件的名称，其中包括当前epoch的编号

                    # 这个字典 state 包含了所有需要保存的信息，以便将来能够恢复训练或进行分析
                    # 一般的ckpt文件是.yaml文件，存放键值对
                    state = {
                        'epoch':epoch,
                        "total_steps": total_steps,
                        'optimizer':optimizer.state_dict(), # 保存优化器的状态，这包括了当前的学习率和其他优化器参数
                        'model':alexnet.state_dict(), # 保存模型的权重和偏置等参数
                        'seed':seed,
                    }
                    torch.save(state,checkpoint_path)









    






        




    