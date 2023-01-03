import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.linspace(-5,5,200)  #linespace 均匀分割
x2 = torch.rand(20)
y_relu = torch.relu(x)
y_sigmoid = torch.sigmoid(x)
y_tanh = torch.tanh(x)

plt.figure(1,figsize=(8,6))  #可连续画好几张图 
plt.subplot(221)  #一张图中画好几个图ijn i行j列第n个
plt.plot(x,y_relu,label='relu',color='red')
plt.ylim(-1,5)

plt.subplot(222)
plt.plot(x,y_sigmoid,label='sigmoid')
plt.ylim(-0.2,1.2)

plt.subplot(223)
plt.plot(x,y_tanh,label='tanh')
plt.ylim(-1.2,1.2)
plt.legend(loc='best',labels='x1')  # handles 一张图中有多个自变量画图,handles=[x1,x2]
                        # loc代表了图例在整个坐标轴平面中的位置（一般选取'best'这个参数值）

plt.show()