import torch

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

#Tensor是pytorch中的基本数据类型
#创建一个Tensor变量w，里面只有一个数据：1.0
w = torch.Tensor([1.0])

#这里表示w需要计算梯度：w.需要_梯度 = True
#Tensor里默认是不会计算关于w的梯度的，所以需要特殊声明
w.requires_grad = True

#计算y^
def forward(x):
    #因为w是Tensor类型的变量，所以这里的 * 已经默认被重载了
    #这里的 * 作用是进行Tensor和Tensor之间的乘法
    #而且这里的x也已经被自动强制类型转换成Tensor类型，所以forward方法返回的是一个Tensor类型的变量
    return x*w

#损失函数
def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y)**2

print("训练前的预测结果:",4,forward(4).item())

for epoch in range(100):

    #这里先把两个数组zip成四个点左边的元组形式并遍历
    for x,y in zip(x_data,y_data):

        #执行损失函数的过程中，就已经在构建计算图了
        l = loss(x,y)

        #这里是从l开始反向传播，逐步计算梯度，最后会存到w里
        #每一次执行完backward之后，pytorch都会释放当前计算图，这样如果每一次迭代的计算图都有差异，也不会影响，很动态，这也是pytorch的核心竞争力
        l.backward()

        #这里输出的时候用到.item()是为了只取出数字，避免构建出计算图
        print('\tloss对于w的梯度:',x,y,w.grad.item())

        #更新权重
        #这里必须加.data，因为张量里的data和grad都是Tensor类型，如果直接拿来计算的话，就不是单纯计算一个数字了，是构建计算图
        w.data = w.data - 0.01 * w.grad.data

        #这里是将w的梯度的数据全都清零，不然下一次再更新梯度的时候，不会自动覆盖，只会在原来基础上叠加，就会出大问题
        w.grad.data.zero_()

    print("迭代次数:",epoch,"loss值:",l.item())

print("训练后的预测结果:",4,forward(4).item())


















