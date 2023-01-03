import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data as Data
import torch.nn.functional as F
import time

a=time.time()
BATCH_SIZE = 250
EPOCH = 20
LR = 0.01
Loss_list=[[], [], [], []]

x = torch.unsqueeze(torch.linspace(-3,3,1000), dim=1)
y = x.pow(2) + torch.normal(torch.zeros(x.size()))

load_data = Data.TensorDataset(x,y)
loader = Data.DataLoader(
    dataset=load_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(1,10)
        self.hidden2 = nn.Linear(10,5)
        self.predict = nn.Linear(5,1)

    def forward(self,x):
        x1 = self.hidden1(x)
        x1 = F.relu(x1)
        x1 = self.hidden2(x1)
        x1 = F.relu(x1)
        x1 = self.predict(x1)
        return x1

net_0 = Net()
net_1 = Net()
net_2 = Net()
net_3 = Net()
net_4 = Net()
net_5 = Net()
net_6 = Net()
net_7 = Net()
net_8 = Net()
net_9 = Net()
nets = [net_0, net_9, net_8, net_7, net_6, net_5, net_4, net_3, net_2, net_1]

optim_0 = torch.optim.SGD(net_0.parameters(), lr=LR)
optim_1 = torch.optim.SGD(net_1.parameters(), momentum=0.1, lr=LR)
optim_2 = torch.optim.SGD(net_2.parameters(), momentum=0.2, lr=LR)
optim_3 = torch.optim.SGD(net_3.parameters(), momentum=0.3, lr=LR)
optim_4 = torch.optim.SGD(net_4.parameters(), momentum=0.4, lr=LR)
optim_5 = torch.optim.SGD(net_5.parameters(), momentum=0.5, lr=LR)
optim_6 = torch.optim.SGD(net_6.parameters(), momentum=0.6, lr=LR)
optim_7 = torch.optim.SGD(net_7.parameters(), momentum=0.7, lr=LR)
optim_8 = torch.optim.SGD(net_8.parameters(), momentum=0.8, lr=LR)
optim_9 = torch.optim.SGD(net_9.parameters(), momentum=0.9, lr=LR)
optims = [optim_0, optim_9, optim_8, optim_7, optim_6, optim_5, optim_4, optim_3, optim_2, optim_1]

loss_func = nn.MSELoss()
for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(loader):
        for net, optim, loss_his in zip(nets, optims, Loss_list):
            pre_y = net(batch_x)
            loss = loss_func(pre_y, batch_y)

            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_his.append(loss)

    print(f'the {epoch}th training is finish')

labels = ['0', '0.9', '0.8', '0.7', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1']
for i, label in enumerate(Loss_list):
    plt.plot(Loss_list[i], label=labels[i])

plt.legend(loc='best')  #加图例
plt.ylabel('loss')
#plt.ylim(-0.05,2)
b = time.time()
print(f'use {b-a} seconds')
plt.show()
