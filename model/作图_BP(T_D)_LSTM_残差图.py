# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 11:38:14 2018

@author: chenc
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from pylab import mpl, matplotlib

from BP_train_predict import BP_Train_Predict
from holiday_def import holidays_generate, date_property_generate


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    # 获取data的列数 columns
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# 设置plt格式
sns.set_style("white")
# 读取数据集导入数据
df = pd.read_excel('.\\data\\gas_load.xlsx')
# 异常数据直接剔除
df.gas_use[601] = (df.gas_use[600]+df.gas_use[602])/2
df.gas_use[614] = (df.gas_use[613]+df.gas_use[615])/2
# 节假日数据生成
holidays = holidays_generate()
date_property = date_property_generate(holidays)
# date_property数据处理，并加入休息日
date_property.ix[date_property.is_holiday==1, 'is_holiday'] = 15
date_property.ix[df.is_holiday==1, 'is_holiday'] = 1
############################################### BP神经网络回归 #####################################################
# 训练数据归一化
train_X = df.ave_temp
train_X = pd.concat((train_X, date_property.is_holiday), axis=1)
train_y = df.gas_use
train_data = pd.concat((train_X, train_y), axis=1)
scaler1 = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler1.fit_transform(train_data)
train_scaled = pd.DataFrame(train_scaled)
# 预测数据归一化
test_X = df.ave_temp
test_X = pd.concat((test_X, date_property.is_holiday), axis=1)
test_y = df.gas_use
test_data = pd.concat((test_X, test_y), axis=1)
scaler2 = MinMaxScaler(feature_range=(0, 1))
test_scaled = scaler2.fit_transform(test_data)
test_scaled = pd.DataFrame(test_scaled)
# 训练 & 预测
pred_y = BP_Train_Predict(train_X=train_scaled[[0,1]], train_y=train_scaled[[2]], input_dim=2, test_X=test_scaled[[0,1]])
pred_data = pd.concat((test_scaled[[0,1]], pd.DataFrame(pred_y)), axis =1)
# 将预测数据反归一化，得回归结果
pred_data = scaler2.inverse_transform(pred_data)
pred_data = pd.DataFrame(pred_data)
# 数据还原
result_hg = pd.DataFrame(columns=('ds', 'is_holiday', 'ave_temp', 'y', 'y_hg', 'hg_err', 'hg_err_rate'))
result_hg.ds = df.date_time
result_hg.is_holiday = date_property.is_holiday
result_hg.ave_temp = df.ave_temp
result_hg.y = df.gas_use
result_hg.y_hg = pred_data[[2]]
# 回归误差=预测-实际
result_hg.hg_err = result_hg.y_hg-result_hg.y
############################################## BP神经网络回归结束 #####################################################

# BP训练结果图与残差图
mpl.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(result_hg.ds[0:850], result_hg.y[0:850], 'k-', lw=0.8, label=u'实际结果')
ax1.plot(result_hg.ds[0:850], result_hg.y_hg[0:850], 'g--', lw=1, label=u'预测结果')
plt.title(u"(a) BPNN预测结果")

ax2 = fig.add_subplot(212)
ax2.plot(result_hg.ds[0:850], result_hg.hg_err[0:850], 'r-', lw=0.8, label=u'残差')
plt.title(u"(b) 残差序列")
ax1.legend(loc='upper right')
ax2.legend(loc='upper right')
plt.show()