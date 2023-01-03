import torch
import torch.nn as nn
import time
import torchvision
import torch.utils.data as Data
import matplotlib.pyplot as plt

a=time.time()
list1=[]
list2=[]
EPOCH = 10
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False  #未下载过为True 下载过False

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Linear(32*7*7, 10, bias=True)  #bias 默认True

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) #  (batch_size, 32, 7, 7)  -->(batch_size, 32*7*7)
        out = self.out(x)
        return out

cnn = CNN()
#print(cnn)
train_data = torchvision.datasets.MNIST(
    root = '/Users/spiderman929147745/Desktop/pytorch_ex/DL/mnist/',  #下载到此路径
    train = True,
    transform = torchvision.transforms.ToTensor(),
    download = DOWNLOAD_MNIST
)

test_data = torchvision.datasets.MNIST(
    root = '/Users/spiderman929147745/Desktop/pytorch_ex/DL/mnist/',
    train= False
)
print(train_data.data.size())  #（60000， 28，28）
print(train_data.targets.size())
plt.imshow(train_data.data[5],cmap='gray')  #显示第6张图
plt.title(train_data.targets[5].numpy())
plt.show()

train_loader = Data.DataLoader(
    dataset= train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# unsqueeze (2000,28,28) -->(2000,1,28,28) /255 (0-255) -->(0-1)
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000]/255
test_y = test_data.targets[:2000]

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

torch.save(cnn,'14CNN.pkl')
test_out = cnn(test_x[:10])
pre_y = torch.max(test_out, 1)[1]
print(pre_y.numpy(), 'prediction number')
print(test_y[:10].numpy(), 'real number')
b=time.time()
print(f'cost {b-a} seconds')
