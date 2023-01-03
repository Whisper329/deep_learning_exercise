from sklearn.utils import shuffle
import torch
import torch.nn as nn
from torch.optim import optimizer
import torch.utils.data as Data
import torch.nn.functional as F
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split  #划分数据集
import matplotlib.pyplot as plt

LOSS=[]
ACCURACY=[]
BATCH_SIZE = 45
LR = 0.01
EPOCH = 20
iris_dataset=load_iris()
# print(iris_dataset.keys())  
# print(iris_dataset['target'].shape)
# X = iris_dataset['data']
# Y = iris_dataset['target']
# data = Data.TensorDataset(X,Y)

# iris_data = Data.DataLoader(
#     dataset=iris_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle= True,
#     num_workers=2
# )

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], test_size=0.1, random_state=1)  #设置random使打乱后的数据集固定不变 别人可以复现
print(X_train.shape)
print(X_test.shape)

# print("X_train:{}".format(X_train[:10]))
# print("y_train:{}".format(y_train[:10]))
#print(y_train.shape)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
#y_train = torch.unsqueeze(y_train, dim=1)
y_test = torch.LongTensor(y_test)
#y_test = torch.unsqueeze(y_test, dim=1)  

#print(y_train.size())

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.h1 = nn.Linear(4,10)
        self.h2 = nn.Linear(10,5)
        self.out = nn.Linear(5,3)

    def forward(self,x):
        x = self.h1(x)
        x = F.relu(x)
        x = self.h2(x)
        pre_y = self.out(x)
        return pre_y


cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()
for epoch in range(EPOCH):
    out = cnn(X_train)
    loss = loss_func(out, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    LOSS.append(loss)

    if epoch%2 == 0:
        pre_y = torch.max(F.softmax(out,dim=0),1)[1]
        pred_y = pre_y.numpy().squeeze()
        target_y = y_train.numpy()
        accuracy_train = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        # print(f'train accuracy is %.4f' %accuracy_train)
        ACCURACY.append(accuracy_train)

print(f'finish training')
print(f'------------------')
out_test = cnn(X_test)
pre_test = torch.max(F.softmax(out_test,dim=0),1)[1]
pred_test = pre_test.numpy()
target_test = y_test.numpy()
accuracy = float((pred_test == target_test).astype(int).sum()) / float(target_test.size)
print(f'test accuracy is %.4f' %accuracy)
print(pred_test)
print(target_test)
plt.plot(ACCURACY)
# plt.savefig('15CNN_ex.jpg')
plt.show()




