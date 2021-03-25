import numpy
import torch

#正常情况下应构造y^ = wx，a应该为2
# 设置初始值为1，经过训练后应逐渐逼近2，而损失函数值应逐渐逼近0
x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]
w = 1.0
#计算y^,即为x*w
def forward(x):
    return x*w

#计算平均损失值
def cost(xs,ys):
    cost = 0
    for x,y in zip(xs,ys):     #传进来的是两个列表x_data和y_data，要用zip()函数打包成点坐标元组列表[(1,2),(2,4),(3,6)]
        y_pred = forward(x)
        cost += (y_pred - y) ** 2   #cost为损失函数加和
    return cost/len(xs)             #返回平均损失值

# 求损失函数对于w的梯度（偏导）平均值
def gradient(xs,ys):
    grad = 0;
    for x,y in zip(xs,ys):
        grad += 2*x*(x*w-y)
    return grad/len(xs)
a = 0.01
print('Predict (before training)',4,forward(4))
for epoch in range(100):
    cost_val = cost(x_data,y_data)
    grad_val = gradient(x_data,y_data)
    w -= a*grad_val
    # 学习率a=0.01是自己设置的，每一步都将系数w-a*梯度（损失函数对于w的梯度）
    # 使得w逐步逼近使得损失值最小的对应数值
    # 使得每次损失值都会逐步逼近最小值，以此达到梯度下降的效果
    print('Epoch:',epoch,'w=',w,'cost=',cost_val)
print('Predict (after training)',4,forward(4))