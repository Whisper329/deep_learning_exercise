# -------
# the 49th loss is 0.7641236186027527


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data as Data

Loss_list = []  #损失函数列表 用于后续绘制损失函数下降曲线
n_x = range(0,100)  #损失函数的横轴

net = nn.Sequential(
    nn.Linear(1,10),
    nn.ReLU(),
    nn.Linear(10,5),
    nn.ReLU(),
    nn.Linear(5,1)
)
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
loss_func = nn.MSELoss()

BATCH_SIZE = 15  #每批训练集的大小
x = torch.unsqueeze(torch.linspace(-5, 5, 30), dim=1)
y = x.pow(2) + 2*torch.rand(x.size())

torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,  #乱序
    num_workers=2  #2个线程/进程
)

for epoch in range(50):
    for step, (batch_x, batch_y) in enumerate(loader):
        prediction = net(batch_x)
        loss = loss_func(prediction, batch_y)

        Loss_list.append(loss)  #添加每次训练的损失函数
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'the {epoch}th loss is {loss}')
    print('-------')   

plt.plot(n_x, Loss_list, '.-', label="Train_Loss")
plt.savefig('10batch_loss.png')
plt.show()
