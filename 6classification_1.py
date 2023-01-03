#这个版本在class net（）中不使用softmax
#在第74行绘图中使用softmax预测 因为loss=CrossEntropyLoss包含softmax

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

n_data = torch.ones(100,2)
#print(n_data.size())
#print(n_data.shape)  #运行后显示torch.size ([100,2])
#n = n_data.numpy()
#print(n.shape)  #运行后显示(100,2 )

x_0 = torch.normal(1*n_data,1) #torch.normal(means, std, out=None) mean均值 std标准差
y_0 = torch.zeros(100)  #标签为0的数据
#print(y_0)
x_1 = torch.normal(-1*n_data,1) #torch.normal(means, std, out=None) mean均值 std标准差
y_1 = torch.ones(100)  #标签为1的数据
#print(x_0.shape)
#print(y_0.shape)

x = torch.cat([x_0,x_1],dim=0).type(torch.FloatTensor)  #按行连接 列数不变 dim默认为0
#print(x.shape)
y = torch.cat([y_0,y_1],dim=0).type(torch.LongTensor)
#y = y.reshape(200,1)  #y的维度为[200,]转为[200,1]
                       #这样需要使用32行的代码画图 squeeze去掉维度为1的 变为[200,]
#print(y.shape)
#data = torch.cat([x,y],dim=1)  #数据拼接 可能没用
#print(data.shape)

#plt.scatter(x[:,0],x[:,1],c=torch.squeeze(y), s=100, lw=0, cmap='RdYlGn')
#plt.scatter(x[:,0],x[:,1],c=y, s=100, lw=0, cmap='RdYlGn')  #c是颜色 s标记大小 lw=线的宽度
#plt.show()

class Net(nn.Module):
    def __init__(self, n_in, n_hi1, n_hi2, n_out):
        super().__init__()
        self.hidden1 = nn.Linear(n_in, n_hi1)
        self.hidden2 = nn.Linear(n_hi1, n_hi2)
        self.out = nn.Linear(n_hi2, n_out)

    def forward(self,x):
        x1 = self.hidden1(x)
        x1 = F.relu(x1)
        x1 = self.hidden2(x1)
        x1 = F.relu(x1)
        x1 = self.out(x1)
        return x1


net = Net(2,12,15,2)
#print(net)
plt.ion()  #开启动态展示图
plt.show()

optimizer = torch.optim.Adam(net.parameters(),lr=0.02)  #规定优化方法
loss_func = nn.CrossEntropyLoss()  #规定损失函数

for t in range(101):  #训练开始
    out = net(x)
    loss = loss_func(out,y)  #计算损失函数
    #print(out.size())
    #print(y.size())

    optimizer.zero_grad()  #梯度清零
    loss.backward()  #反向传播
    optimizer.step()  #优化参数

    if t%5 == 0:
        print(f'the {t}th training loss is {loss}')
        print('--------------------')

        plt.cla()
        # 过了一道 softmax 的激励函数后的最大概率才是预测值
        prediction = torch.max(F.softmax(out,dim=0), 1)[1]            
        pred_y = prediction.numpy().squeeze()
        target_y = y.numpy()
        plt.scatter(x.numpy()[:, 0], x.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/200.  # 预测中有多少和真实值一样
        plt.text(1, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.3)

plt.ioff()  # 停止画图
plt.show()
