import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-3,3,200),dim=1)  #unsqueeze dim=1 扩充列维度 把1*200的数据转为200*1
#x = x.reshape(200,1)  #和unsqueeze一样 改变维度
y = x.pow(2) + 1.5*torch.rand(x.size())  #随即加入噪声
#plt.scatter(x,y)  #绘制散点图
#plt.show()
x_test = torch.unsqueeze(torch.randn(100),dim=1)


net = nn.Sequential(
    nn.Linear(1,10),
    nn.ReLU(),
    nn.Linear(10,1)
)
#print(net)
plt.ion()  #开启动态展示图
plt.show()
optimizer = torch.optim.Adam(net.parameters(),lr=0.15)  #规定优化方法
loss_func = nn.MSELoss()  #规定损失函数

for t in range(100):  #训练开始
    prediction = net(x)
    loss = loss_func(prediction,y)  #计算损失函数
 
    optimizer.zero_grad()  #梯度清零
    loss.backward()  #反向传播
    optimizer.step()  #优化参数

pre = net(x)

torch.save(net,'9save_net.pkl')
torch.save(net.state_dict(), '9save_para.pkl')
#只存参数的时候要注意在提取时需构建完全相同的网络，然后再传入参数 42-46行

net2 = torch.load('9save_net.pkl')
pre2 = net2(x_test)

net3 = nn.Sequential(
    nn.Linear(1,10),
    nn.ReLU(),
    nn.Linear(10,1)
)
net3.load_state_dict(torch.load('9save_para.pkl'))
pre3 = net3(x_test)


plt.figure(1,figsize=(15,6)) 
plt.subplot(131)
plt.scatter(x, y.detach().numpy())  #绘制散点图
plt.title('net')
plt.plot(x, prediction.detach().numpy(), 'r-', lw=5)

plt.subplot(132)
plt.scatter(x_test, pre2.detach().numpy())  #绘制散点图
plt.title('save_net')
plt.plot(x, prediction.detach().numpy(), 'r-', lw=5, alpha=0.2)

plt.subplot(133)
plt.scatter(x_test, pre3.detach().numpy())  #绘制散点图
plt.plot(x, prediction.detach().numpy(), 'r-', lw=5, alpha=0.2)
plt.title('save_para')
plt.savefig('9save.png')  #保存图片 在plt.show()之前保存
plt.show(5)  