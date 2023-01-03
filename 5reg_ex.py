import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.linspace(-2,2,200)
x = x.reshape(200,1)
y = x.pow(3) + torch.rand(x.size())
#plt.scatter(x,y)
#plt.plot(x,y)
#plt.show()
plt.ion()
plt.show()

class Net(nn.Module):
    def __init__(self, n_in, n_hidden1, n_hidden2, n_out):
        super().__init__()
        self.hidden1 = nn.Linear(n_in, n_hidden1)
        self.hidden2 = nn.Linear(n_hidden1, n_hidden2)
        self.predict = nn.Linear(n_hidden2, n_out)

    def forward(self, x):
        x1 = self.hidden1(x)
        x2 = torch.relu(x1)
        x3 = self.hidden2(x2)
        x4 = self.predict(x3)
        return x4

net = Net(1,15,10,1)

optimizer = torch.optim.Adam(net.parameters(), lr=0.05)
loss_func = nn.MSELoss()

for t in range(200):
    prediction = net(x)
    loss = loss_func(y, prediction)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t%5 == 0:
        #print(f'the {t}th training loss is {loss}')
        #print('----------------------------')
        plt.cla()  
        plt.scatter(x,y)
        plt.plot(x, prediction.detach().numpy(), 'r-',lw=5)
        plt.text(0.5, 0, 'Loss=%.5f' % loss.detach().numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()  #使最后一张图时保留
