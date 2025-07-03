import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 定义与训练相同的模型结构
class Net(torch.nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = self.l5(x)
        return x
    
# 加载模型
def load_model():
    model = Net()
    model.load_state_dict(torch.load("mnist_model.pth"))
    model.eval()
    # 打印部分权重，确认加载是否正确
    for param in model.parameters():
        print(param.data)
        break  # 只打印一层

    return model

# 预测单张图片
def predict_image(image_path,model):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)
    
    # 图片转为灰度图，调整大小为28*28
    img = Image.open(image_path).convert('L')
    img = img.resize((28,28))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)

  # 调试信息：打印图像张量的最小值和最大值，确保在 [0, 1] 范围内
    print("Image Tensor - min:", img_tensor.min().item(), "max:", img_tensor.max().item())


    # 使用模型预测
    with torch.no_grad():
        output = model(img_tensor)
        _, prediction = torch.max(output,1)  # 找预测的概率最大值，即就是属于哪个数字

    return prediction.item()

if __name__ == '__main__':

    # 先加载模型
    model = load_model()

    image_path = 'IMG_37E5DA0A1050-1.jpeg'
    prediction_lablel = predict_image(image_path,model)
    print("Predicted label:",prediction_lablel)