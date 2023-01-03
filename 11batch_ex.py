import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data as Data

Loss_list = []
BATCH_SIZE = 40
#n_x = range(150)  plt.plot的x轴 可不用定义 

x = torch.unsqueeze(torch.linspace(-3,3,400), dim=1)
y = x.pow(2) + 1.5*torch.normal(torch.zeros(x.size()))

net = nn.Sequential(
    nn.Linear(1,20),
    nn.ReLU(),
    nn.Linear(20,5),
    nn.ReLU(),
    nn.Linear(5,1)
)
optimizer = torch.optim.Adam(net.parameters(), lr=0.02)
loss_func = nn.MSELoss()

load_data = Data.TensorDataset(x,y)
loader = Data.DataLoader(
    dataset=load_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

for epoch in range(30):
    for step, (batch_x, batch_y) in enumerate(loader):
        pre_y = net(batch_x)
        loss = loss_func(pre_y, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        Loss_list.append(loss)
        print(f'the {epoch*10 + step + 1}th loss is {loss}')
        print('-------')

plt.plot(range(len(Loss_list)),Loss_list)  #可以不指定x x轴默认为range(len(y))
plt.savefig('11batch_ex.jpg')
plt.show()
