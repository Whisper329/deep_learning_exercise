import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.linspace(-3,3,200)
x.reshape(200,1)
y = x.pow(2) + 1.5*torch.rand(x.size())

plt.scatter(x,y)  #绘制散点图
plt.plot(x,y)  #绘制连续图
plt.show()  #上述两张图合为一张图