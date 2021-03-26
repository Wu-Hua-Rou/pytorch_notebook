import torch

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])

#创建一个类，继承torch.nn.Module这个类
class LinearModel(torch.nn.Module):
    #构造函数
    def __init__(self):
        super(LinearModel, self).__init__()
        #这里括号里的1,1表示输入和输出都是一维的
        #定义一个线性计算块
        self.linear = torch.nn.Linear(1, 1)

    #这里必须叫这个名字来重写父类的forward方法
    #实现前馈的计算
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

    #这里不需要写backward，因为Model类创建的对象会自动执行反向传播

model = LinearModel()

#构造损失函数，不求均值，此方法需要两个参数，y^和y
#这里损失函数：loss = (y - y^)**2
criterion = torch.nn.MSELoss(size_average=False)

#优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    #计算y^
    y_pred = model(x_data)
    #损失函数，输入y^和y计算损失值
    loss = criterion(y_pred, y_data)
    #每纠正一次，输出一次损失值
    print(epoch, loss.item())

    #每次把梯度归零，不能省略！！！
    optimizer.zero_grad()
    #反向传播
    loss.backward()
    #反向传播之后进行更新权重
    optimizer.step()

#打印权重和偏移量
print('w=',model.linear.weight.item())
print('h=',model.linear.bias.item())

#模型测试
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ',y_test.data.item())
