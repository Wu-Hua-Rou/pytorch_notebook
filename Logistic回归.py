import torch
import torch.nn.functional as F
#torch.nn.functional包里定义了sigmoid函数（这里即是Logistic函数：1/(1+e^-x)）

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
# 线性回归y值：y_data = torch.Tensor([[2.0], [4.0], [6.0]])，注意不同点！
y_data = torch.Tensor([[0], [0], [1]])
#由于要做分类，所以y值只需要有一个1，剩余全为0

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x):
        # 线性回归代码：y_pred = self.linear(x)
        y_pred = F.sigmoid(self.linear(x))
        # 这里和普通线性回归不同，这里算出y^后要带入Logistic函数，过滤成0-1之间的数
        return y_pred
model = LinearModel()
# criterion = torch.nn.MSELoss(size_average=False)
criterion = torch.nn.BCELoss(size_average=False)
#这里和普通线性回归不同，这里要使用BCEloss损失函数，即为二分类交叉熵函数：loss = -(y*logy^ + (1-y)*log(1-y^))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(10000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print('w=',model.linear.weight.item())
print('h=',model.linear.bias.item())
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ',y_test.data.item())
