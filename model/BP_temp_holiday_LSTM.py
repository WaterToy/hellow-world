# -*- coding: utf-8 -*-
"""
Created on Wed Mar  17 11:38:14 2018

@author: chenc
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import *  
from numpy import concatenate
from keras.optimizers import SGD
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
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

def regr_pred(input_array, hist_days, pred_days, n_train_days):
    global inv_yhat_transpose, inv_y_transpose
    # input_array = input_array.astype('float32')
    # 转化为有监督数据
    re_frame = series_to_supervised(input_array, hist_days, pred_days)
    test= pd.DataFrame(list(re_frame.iloc[-1][-hist_days:])+list(np.random.randint(100000,150000,pred_days))).T.values        # 后七位为无效数据
    re_frame = pd.DataFrame(concatenate((re_frame,test), axis=0))
    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    reframed = scaler.fit_transform(re_frame)
    
    # 划分训练数据与测试数据
    train = reframed[:len(input_array)-pred_days-hist_days+1, :]
    test = reframed[-1, :]
    test = pd.DataFrame(test).T.values
    # 拆分输入与输出
    train_x, train_y = train[:, :-pred_days], train[:, -pred_days:]
    test_x = test[:, :-pred_days]
    # reshape input to be 3D [samples, timesteps, features]
    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(Dense(pred_days))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_x, train_y, epochs=1000, batch_size=40, verbose=0, shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    # make a prediction
    yhat = model.predict(test_x)
    test_x = test_x.reshape((test_x.shape[0], test_x.shape[2]))
    # invert scaling for forecast
    inv_yhat = concatenate((test_x, yhat), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,-pred_days:]
    # 转置
    inv_yhat_transpose = pd.DataFrame([i for i in range(inv_yhat.shape[1])])
    for i in range(inv_yhat.shape[0]):
        inv_yhat_transpose = pd.concat((inv_yhat_transpose, pd.DataFrame(list(inv_yhat[i]))), axis=1)
    inv_yhat_transpose.columns = [i for i in range(-1, inv_yhat_transpose.shape[1]-1, 1)]
    inv_yhat_transpose.drop(inv_yhat_transpose.columns[[0]], axis=1, inplace=True)
    return inv_yhat_transpose


# 参数设置
EEMD_start_days = 0
EEMD_end_days = 851
LSTM_train_days = EEMD_end_days-EEMD_start_days
LSTM_learn_days = 45    # 学习的天数
LSTM_pred_days = 15     # 预测的天数
# 设置plt格式
sns.set_style("white")
# 读取数据集导入数据
df = pd.read_excel('D:\\Mission\\云同步\\学习\\20180124_小论文\\20171123数据整理\武汉\居&商_武汉20150101~20170823.xlsx')
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
train_scaled = pd.DataFrame(train_scaled)    # 转换为 DataFrame 格式
# 预测数据归一化
test_X = df.ave_temp
test_X = pd.concat((test_X, date_property.is_holiday), axis=1)
test_y = df.gas_use
test_data = pd.concat((test_X, test_y), axis=1)
scaler2 = MinMaxScaler(feature_range=(0, 1))
test_scaled = scaler2.fit_transform(test_data)
test_scaled = pd.DataFrame(test_scaled)    # 转换为 DataFrame 格式
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
result_hg.hg_err = result_hg.y_hg-result_hg.y    # 回归误差=预测-实际
result_hg.hg_err_rate = abs(result_hg.hg_err/result_hg.y)
hg_err_mean = result_hg.hg_err_rate[EEMD_end_days:(EEMD_end_days+LSTM_pred_days)].mean()
plt.plot([i for i in range(LSTM_pred_days)], result_hg.y[EEMD_end_days:(EEMD_end_days+LSTM_pred_days)], 'r')
plt.plot([i for i in range(LSTM_pred_days)], result_hg.y_hg[EEMD_end_days:(EEMD_end_days+LSTM_pred_days)], 'g')
plt.show()
############################################## BP神经网络回归结束 #####################################################

################################################# LSTM神经网络 #######################################################
LSTM_df = result_hg.hg_err[EEMD_start_days:EEMD_end_days]
inv_yhat_transpose = regr_pred(list(LSTM_df.values), LSTM_learn_days, LSTM_pred_days, LSTM_train_days)
plt.plot([i for i in range(LSTM_pred_days)], result_hg.hg_err[EEMD_end_days:(EEMD_end_days+LSTM_pred_days)], 'r')        #误差实际值
plt.plot([i for i in range(LSTM_pred_days)], inv_yhat_transpose[0], 'g')        #误差预测值
plt.show()

y_pri = result_hg.y[EEMD_end_days:(EEMD_end_days+LSTM_pred_days)].reset_index(drop=True)
y_zh = result_hg.y_hg[EEMD_end_days:(EEMD_end_days+LSTM_pred_days)].reset_index(drop=True)-inv_yhat_transpose[0]
plt.plot([i for i in range(LSTM_pred_days)], y_pri, 'r')        # y实际
plt.plot([i for i in range(LSTM_pred_days)], y_zh, 'g')        # y综合预测值
plt.show()
print("BP误差率：", hg_err_mean)
print("综合预测误差率：", (abs(y_zh-y_pri)/y_pri).mean())

############################################## LSTM回归预测结束 #####################################################