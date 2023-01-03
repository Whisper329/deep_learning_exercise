import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

n_data = torch.ones(200, 2)
x0 = torch.normal(2*n_data, 1.5)
y0 = torch.zeros(200)
x1 = torch.normal(-2*n_data, 1.5)
y1 = torch.ones(200)

x = torch.cat([x0, x1], dim=0).type(torch.FloatTensor)
y = torch.cat([y0, y1], dim=0).type(torch.LongTensor)
#y = y.reshape(200,1)
#print(y.size())

net = nn.Sequential(
    nn.Linear(2,10),
    nn.ReLU(),
    nn.Linear(10,5),
    nn.ReLU(),
    nn.Linear(5,2)
)

plt.ion()
plt.show()

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()

for t in range(101):
    out = net(x)
    #print(out.size())
    #print(y.size())
    loss = loss_func(out, y)
    ########loss_func(out, y) 一定是out前 y后

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t%2 == 0:
        prediction = torch.max(F.softmax(out),1)[1]
        pre_y = prediction.numpy().squeeze()
        target_y = y.numpy()
        accuracy = sum(pre_y == target_y)/400.
        plt.cla()
        plt.scatter(x.numpy()[:,0], x.numpy()[:,1], c=pre_y, s=50, lw=2, cmap='summer')
        plt.text(1, -3, 'accuracy=%.3f' % accuracy, fontdict={'size':20, 'color':'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
