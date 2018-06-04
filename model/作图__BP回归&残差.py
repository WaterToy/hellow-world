import pymysql
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fbprophet import Prophet
from keras.optimizers import SGD
# Dense全连接层
from keras.layers import Dense,Activation
# 按顺序构成的模型
from keras.models import Sequential

##########################################################################################################
#                                       神经网络回归 + fb开源工具预测结果
##########################################################################################################
# 设置plt格式
sns.set_style("white")  
# 导入输入、输出值
# 读取数据集导入数据
df = pd.read_excel('.\\data\\gas_load.xlsx')
tmp = df.loc[df.is_holiday==0][['date_time', 'is_holiday', 'gas_use', 'high_temp', 'low_temp', 'ave_temp']]
df_no_holiday = tmp.loc[df.gas_use!=36391.8].loc[df.gas_use!=3261750.0]
x_data = df_no_holiday.ave_temp
y_data_max = df_no_holiday.gas_use.max()
y_data_min = df_no_holiday.gas_use.min()
y_data = 0.7*(df_no_holiday.gas_use-y_data_min)/(y_data_max-y_data_min)+0.15


# 加激活函数的方法1：mode.add(Activation(''))
np.random.seed(0)

# 构建一个顺序模型
model = Sequential()

# 在模型中添加一个全连接层
# units是输出维度,input_dim是输入维度(shift+两次tab查看函数参数)
# 输入1个神经元,隐藏层10个神经元,输出层1个神经元
model.add(Dense(units=12, input_dim=1))
model.add(Activation('tanh'))   # 增加非线性激活函数
model.add(Dense(units=3, activation='tanh'))   # 增加非线性激活函数
model.add(Dense(units=1))   # 默认连接上一层input_dim=10
model.add(Activation('tanh'))

# 定义优化算法(修改学习率)
defsgd = SGD(lr=0.1)

# 编译模型
model.compile(optimizer=defsgd, loss='mse')   # optimizer参数设置优化器,loss设置目标函数

# 训练模型
for step in range(20000):
    # 每次训练一个批次
    cost = model.train_on_batch(x_data, y_data)
    # 每500个batch打印一个cost值
    if step % 1000 == 0:
        print('cost:', cost)

# 打印权值和偏置值
W, b = model.layers[0].get_weights()   # layers[0]只有一个网络层
print('W:', W, 'b:', b)

# x_data输入网络中，得到预测值y_pred
y_pred_all = model.predict(df.ave_temp)

# 作图函数移至最后
# plt.scatter(df_no_holiday.ave_temp, 0.7*(df_no_holiday.gas_use-y_data_min)/(y_data_max-y_data_min)+0.15)

# plt.plot(df.ave_temp, y_pred_all, 'r-', lw=3)
# plt.show()


# 数据还原
result_hg = pd.DataFrame(columns=('ds', 'is_holiday', 'ave_temp', 'y', 'y_hg', 'err_hg', 'err_rate_hg', 'tmp_err_hg'))
result_hg.ds = df.date_time
result_hg.is_holiday = df.is_holiday
result_hg.ave_temp = df.ave_temp
result_hg.y = df.gas_use
result_hg.y_hg = (y_pred_all[:, 0] - 0.15)/0.7 * (y_data_max-y_data_min) + y_data_min
result_hg.err_hg = result_hg.y_hg - result_hg.y    # 预测-实际
result_hg.err_rate_hg = abs(result_hg.err_hg / result_hg.y)
result_hg.tmp_err_hg = 0.7*(result_hg.err_hg -result_hg.err_hg.min())/ (result_hg.err_hg.max() - result_hg.err_hg.min()) +0.15
# y_pred_all_data = y_pred_all[:, 0] * 6000000
# hg_err = y_pred_all_data - y_all_data
# hg_err_rate = abs(hg_err / y_all_data)
hg_err_mean = result_hg.err_rate_hg.mean()


# 作图部分
from pylab import *  
mpl.rcParams['font.sans-serif'] = ['SimHei']  
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.scatter(df_no_holiday.ave_temp, 0.7*(df_no_holiday.gas_use-y_data_min)/(y_data_max-y_data_min)+0.15, label=u'归一化后的燃气日负荷')
ax1.plot(df.ave_temp, y_pred_all, 'r-', lw=3, label=u'预测结果')

ax2 = fig.add_subplot(212)
new = 0.7 * (df.gas_use.values-y_data_min)/(y_data_max-y_data_min)+0.15
ax2.plot(y_pred_all[:, 0]-new, label=u'残差')
y3 = df.is_holiday
y4 = y3.loc[y3==2]
ax2.scatter(y4.index, -0.15*y4, c='r', marker='^', label=u'节假日')

ax1.legend(loc='upper right')
ax2.legend(loc='upper right')

plt.show()
