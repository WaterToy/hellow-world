# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 20:12:14 2018

@author: chenc
"""
import numpy as np
from keras.optimizers import SGD
from keras.layers import Dense,Activation
from keras.models import Sequential

def BP_Train_Predict(train_X, train_y, input_dim, test_X, train_steps=20000, learn_rate=0.1):
    print('***************神经网络回归开始***************')
    np.random.seed(0)
    # 构建一个顺序模型
    model = Sequential()
    # 输入2个神经元,隐藏层12个神经元,输出层1个神经元
    model.add(Dense(units=12, input_dim=input_dim))
    model.add(Activation('tanh'))   # 增加非线性激活函数
    model.add(Dense(units=2, activation='tanh'))   # 增加非线性激活函数
    model.add(Dense(units=1))   # 默认连接上一层input_dim=12
    model.add(Activation('tanh'))
    # 定义优化算法(修改学习率)
    defsgd = SGD(lr=learn_rate)
    # 编译模型
    model.compile(optimizer=defsgd, loss='mse')   # optimizer参数设置优化器,loss设置目标函数
    # 训练模型
    for step in range(train_steps):
        # 每次训练一个批次
        cost = model.train_on_batch(train_X, train_y)
        # 每500个batch打印一个cost值
        if step % 1000 == 0:
            print('cost:', cost)
    # 打印权值和偏置值
    W, b = model.layers[0].get_weights()   # layers[0]只有一个网络层
    print('W:', W, 'b:', b)
    
    # x_data输入网络中，得到预测值y_pred
    y_pred = model.predict(test_X)
    return y_pred