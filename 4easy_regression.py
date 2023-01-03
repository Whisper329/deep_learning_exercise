import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-3,3,200),dim=1)  #unsqueeze dim=1 扩充列维度 1维数据变为2维数据
#x = x.reshape(200,1)  #和unsqueeze一样 改变维度
y = x.pow(2) + 1.5*torch.rand(x.size())  #随即加入噪声
#plt.scatter(x,y)  #绘制散点图
#plt.show()


class Net(nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super().__init__()
        self.hidden = nn.Linear(n_input,n_hidden)
        self.predict = nn.Linear(n_hidden,n_output)

    def forward(self,x_in):
        x1 = self.hidden(x_in)
        x2 = F.relu(x1)
        x3 = self.predict(x2)
        return x3

net = Net(1,10,1)
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

    if t%5 == 0:
        print(f'the {t}th training loss is {loss}')
        print('--------------------')
        plt.cla()  #消除上一次动态曲线的痕迹
        plt.scatter(x,y)
        plt.plot(x,prediction.detach().numpy(),'r-',lw=5)
        plt.text(0.5, 0, 'Loss=%.5f' % loss.detach().numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)
        
plt.ioff()
plt.show()  #最后一张图片保留