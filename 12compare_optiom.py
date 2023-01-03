import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import time

a = time.time()
BATCH_SIZE = 250
EPOCH = 40
LR = 0.01

x = torch.unsqueeze(torch.linspace(-3, 3, 1000), dim=1)
y = x.pow(2) + torch.normal(torch.zeros(x.size()))
# y = torch.normal(torch.zeros(x.size()))
y1 = x.pow(2) + 0.2*torch.randn(x.size())
# y2 = torch.rand(x.size())

# plt.figure(1,figsize=(15,8))
# plt.subplot(131)
# plt.scatter(x, y)
# plt.title('normal')

# plt.subplot(132)
# plt.scatter(x, y1)
# plt.title('randn')

# plt.subplot(133)
# plt.scatter(x, y2)
# plt.title('rand')
# plt.savefig('12normal_randn_rand')
# plt.show()
load_data = Data.TensorDataset(x, y1)
loader = Data.DataLoader(
    dataset=load_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(1, 20)   # hidden layer
        self.hidden2 = nn.Linear(20, 10)
        self.predict = nn.Linear(10, 1)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden1(x))      # activation function for hidden layer
        x = F.relu(self.hidden2(x)) 
        x = self.predict(x)             # linear output
        return x

net_SGD = Net()
net_Monentum = Net()
net_Adam = Net()
net_RMSprop = Net()
nets = [net_SGD, net_Monentum, net_Adam, net_RMSprop]

optim_SGD = torch.optim.SGD(net_SGD.parameters(),lr=LR)
optim_Monentum = torch.optim.SGD(net_Monentum.parameters(),lr=LR, momentum=0.8)
optim_Adam = torch.optim.Adam(net_Adam.parameters(),lr=LR)
optim_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(),lr=LR, alpha=0.9)
optimizers = [optim_SGD, optim_Monentum, optim_Adam, optim_RMSprop]

loss_func = nn.MSELoss()
loss_list = [[], [], [], []]

for epoch in range(EPOCH):
    print('Epoch: ', epoch)
    for step, (b_x, b_y) in enumerate(loader):

        # 对每个优化器, 优化属于他的神经网络
        for net, opt, l_his in zip(nets, optimizers, loss_list):
            output = net(b_x)              # get output for every net
            loss = loss_func(output, b_y)  # compute loss for every net
            opt.zero_grad()                # clear gradients for next train
            loss.backward()                # backpropagation, compute gradients
            opt.step()                     # apply gradients
            l_his.append(loss)             # loss recoder

labels = ['SGD', 'Momentum', 'Adam', 'RMSprop']
for i, label in enumerate(loss_list):
    plt.plot(loss_list[i], label=labels[i])

plt.legend(loc='best')  #加图例
plt.ylabel('loss')
#plt.ylim(-0.05,2)
b = time.time()
print(f'costs {b-a} seconds')
plt.show()