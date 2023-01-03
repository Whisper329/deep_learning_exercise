#torchvision.tranforms.ToTensor没用 nparray无法转成tensor
#(5000, 32, 32, 3)应该变为(5000, 3, 32, 32)

import torch
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as Data
import torch.nn as nn

BATCH_SIZE = 500
EPOCH = 2
LR = 0.01
DOWNLOAD_CIFAR = False

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = torchvision.datasets.CIFAR10(
    root = './cifar10/',  #下载到此路径
    train = True,
    transform = torchvision.transforms.ToTensor(),
    download = DOWNLOAD_CIFAR
)
test_data= torchvision.datasets.CIFAR10(
    root='./cifar10/',
    train=False, 
    download=DOWNLOAD_CIFAR,
    transform = transform,
)
# print(test_data.data.shape)  #(10000, 32, 32, 3)
train_loader = Data.DataLoader(
    dataset= train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

test_loader = Data.DataLoader(
    dataset= test_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)
# print(train_data.classes) #输出分类
# print(train_data.targets[:10]) #输出分类标签
# print(type(train_data.targets)) #打印数据类型
#print(train_data.data.shape)  #输出train_data数据维度  (50000, 32, 32, 3)
# print(train_data.targets.size())
# 展示第一张图片
# plt.imshow(train_data.data[0])
# plt.title(train_data.targets[0])
# plt.show()

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 5, 1, 2),  #(8, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(2)  #(8, 16, 16)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, 1, 1), #(16, 16, 16)
            nn.ReLU(),
            nn.AvgPool2d(2)  #(16, 8, 8)
        )
        self.out = nn.Linear(16*8*8, 10)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) #  (batch_size, 16, 8, 8)  -->(batch_size, 16*8*8)
        out = self.out(x)
        return out

cnn = CNN()

test_x = test_data.data[:500]
test_y = test_data.targets[:500]

test_x = torch.FloatTensor(test_x)
test_y = torch.FloatTensor(test_y)

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   
loss_func = nn.CrossEntropyLoss()   

for epoch in range(EPOCH):
    print(f'{epoch}th training is start')
    for step, (b_x, b_y) in enumerate(train_loader):
        out = cnn(b_x)
        loss = loss_func(out, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step%50 == 0:
            test_out = cnn(test_x)
            pre_y = torch.max(test_out, 1)[1]
            accuracy = sum(pre_y == test_y)/test_y.size(0)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

test_out = cnn(test_x[:10])
pre_y = torch.max(test_out, 1)[1]
print(pre_y.numpy(), 'prediction number')
print(test_y[:10].numpy(), 'real number')
