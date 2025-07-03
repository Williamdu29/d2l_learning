import torch

x_data = [1,2,3]
y_data = [2,4,6]

W = torch.Tensor([1.0])
W.requires_grad = True

# 构建计算图

def forward(x):
    return x * W

def loss(x,y):
    y_pred = forward(x)
    loss = (y_pred-y)**2
    return loss

lr = 0.01

for epoch in range(100):
    for x,y in zip(x_data,y_data):
        l = loss(x,y)
        l.backward()
        print("\tgrad:",x,y,W.grad.item())

        W.data -= lr*W.grad.data
        W.grad.data.zero_()
    print("progresses:",epoch,l.item())

print("predict after training:",4,forward(4).item())

