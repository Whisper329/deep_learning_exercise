
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-3,3,100),dim=1)
y = x.pow(2) + torch.rand(x.size())

net = nn.Sequential(
    nn.Linear(1,10),
    nn.ReLU(),
    nn.Linear(10,1)
)
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)
loss_func = nn.MSELoss()

plt.ion()
plt.show()

for t in range(201):
    prediction = net(x)
    loss = loss_func(y, prediction)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t%4 == 0:
        # print(f'the {t}th loss is {loss}')
        # print('-------')

        plt.cla()
        plt.scatter(x,y)
        plt.plot(x,prediction.detach().numpy(), 'r-', lw=5)  #lw线的粗细
        plt.text(1, 0, 'Loss=%.3f' %loss.detach().numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.text(0, 8, 'lr=0.02', fontdict={'size':20, 'color':'blue'})
        plt.pause(0.1)

plt.ioff()
plt.show()

