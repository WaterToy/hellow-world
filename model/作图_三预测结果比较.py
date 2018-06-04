# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 11:38:14 2018

@author: chenc
"""
import math
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

# 参数设置
LSTM_train_days = 900   # 预测的输出从 LSTM_train_days+30 开始
LSTM_learn_days = 30    # 学习的天数
LSTM_pred_days = 15     # 预测的天数，一旦修改，需对应修改series_to_supervised()结果删除的列数
LSTM_test_days = 801    # 测试集从此开始
# 设置plt格式
sns.set_style("white")
# 读取数据集导入数据
df = pd.read_excel('D:\\20180124_小论文\\20171123数据整理\武汉\居&商_武汉20150101~20170823.xlsx')
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
result_hg.hg_err_rate = result_hg.hg_err/result_hg.y
hg_err_mean = result_hg.hg_err_rate.mean()

# plt.plot(result_hg.ds,result_hg.hg_err)
# 作图展示real与pred
# plt.plot(result_hg.ds, result_hg.y, label='real')
# plt.plot(result_hg.ds, result_hg.y_hg, label='pred')
# plt.legend(loc=0)
# plt.show()
############################################## BP神经网络回归结束 #####################################################

################################################# LSTM神经网络 #######################################################
LSTM_df = pd.concat((date_property[['is_holiday']], result_hg[['hg_err']]), axis=1)
LSTM_df_values = LSTM_df.values.astype('float32')
# 转化为有监督数据
re_frame = series_to_supervised(LSTM_df_values, LSTM_learn_days, LSTM_pred_days)
# 删除不预测的列
re_frame.drop(re_frame.columns[[60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88]], axis=1, inplace=True)
# 归一化
scaler3 = MinMaxScaler(feature_range=(0, 1))
scaled3 = scaler3.fit_transform(re_frame)
# 训练数据与测试数据划分
LSTM_train = scaled3[:LSTM_train_days, :]
LSTM_test = scaled3[LSTM_test_days:, :]    # 对应于 result_hg.loc[800:921],2017.3.11 和 2017.7.10
# 拆分输入输出
LSTM_train_X, LSTM_train_y = LSTM_train[:, :-LSTM_pred_days], LSTM_train[:, -LSTM_pred_days:]
LSTM_test_X, LSTM_test_y = LSTM_test[:, :-LSTM_pred_days], LSTM_test[:, -LSTM_pred_days:]

# reshape input to be 3D [samples, timesteps, features]
LSTM_train_X = LSTM_train_X.reshape((LSTM_train_X.shape[0], 1, LSTM_train_X.shape[1]))
LSTM_test_X = LSTM_test_X.reshape((LSTM_test_X.shape[0], 1, LSTM_test_X.shape[1]))
# design network
model = Sequential()
model.add(LSTM(120, input_shape=(LSTM_train_X.shape[1], LSTM_train_X.shape[2])))
model.add(Dense(LSTM_pred_days))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(LSTM_train_X, LSTM_train_y, epochs=1000, batch_size=45, validation_data=(LSTM_test_X, LSTM_test_y), verbose=2, shuffle=False)
# plot history
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# plt.show()
# make a prediction
LSTM_yhat = model.predict(LSTM_test_X)
LSTM_test_X = LSTM_test_X.reshape((LSTM_test_X.shape[0], LSTM_test_X.shape[2]))
# invert scaling for forecast
LSTM_inv_yhat = np.concatenate((LSTM_test_X, LSTM_yhat), axis=1)
LSTM_inv_yhat = scaler3.inverse_transform(LSTM_inv_yhat)
LSTM_inv_yhat = LSTM_inv_yhat[:,-LSTM_pred_days:]    # 对应实际行数从 LSTM_test_days+30 行开始
# 转置LSTM_inv_yhat
LSTM_inv_yhat = pd.DataFrame(LSTM_inv_yhat)
LSTM_result = pd.DataFrame(LSTM_inv_yhat.iloc[0])
for i in range(LSTM_inv_yhat.shape[0]-1):
    LSTM_result = pd.concat((LSTM_result, pd.DataFrame(LSTM_inv_yhat.iloc[i+1])), axis=1)    # 起始时间 对应于 result_hg.loc[800:921]

# 利用LSTM_result还原最终预测结果
BP_y_hg = result_hg.y_hg.loc[LSTM_test_days+LSTM_learn_days:]    # 921~935为最后一个预测的行区间
y_hg_date = result_hg.ds.loc[LSTM_test_days+LSTM_learn_days:]    # 拎出对应回归结果的时间节点
BP_LSTM_result = pd.DataFrame([i for i in range(LSTM_pred_days)])
for i in range(LSTM_result.shape[1]):
    BP_LSTM_item = BP_y_hg.loc[LSTM_test_days+LSTM_learn_days+i:LSTM_test_days+LSTM_learn_days+LSTM_pred_days-1+i].values-LSTM_result[i].values
    BP_LSTM_result = pd.concat((BP_LSTM_result, pd.DataFrame(BP_LSTM_item)), axis=1)
BP_LSTM_result.columns = [i for i in range(-1, BP_LSTM_result.shape[1]-1, 1)]
BP_LSTM_result.drop(BP_LSTM_result.columns[[0]], axis=1, inplace=True)

# 寻找结果对应的原始数据
primary_y = pd.DataFrame([i for i in range(LSTM_pred_days)])
for i in range(LSTM_result.shape[1]):
    primary_y_item = result_hg.y.loc[LSTM_test_days+LSTM_learn_days+i:LSTM_test_days+LSTM_learn_days+LSTM_pred_days-1+i].values
    primary_y = pd.concat((primary_y, pd.DataFrame(primary_y_item)), axis=1)
primary_y.columns = [i for i in range(-1, primary_y.shape[1]-1, 1)]
primary_y.drop(primary_y.columns[[0]], axis=1, inplace=True)

# 计算误差
BP_LSTM_err_rate = []
for i in range(primary_y.shape[1]):
    BP_LSTM_err = (abs(BP_LSTM_result[i] - primary_y[i])/primary_y[i]).mean()
    BP_LSTM_err_rate.append(BP_LSTM_err)
################################################# LSTM神经网络结束 #######################################################

################################################ 单独的LSTM神经网络 ######################################################
df_lstm_alone = df[['ave_temp', 'is_holiday', 'gas_use']]
# 补充日期序号
day_index= [i%365 for i in range(len(df.gas_use))]
df_lstm_alone.insert(0, 'day_index', day_index)
df_lstm_alone_values = df_lstm_alone.values.astype('float32')
# 转化为有监督数据
lstm_alone_re_frame = series_to_supervised(df_lstm_alone_values, 30, 15)
# 删除不预测的列
lstm_alone_re_frame.drop(lstm_alone_re_frame.columns[[120,121,122,
                                124,125,126,
                                128,129,130,
                                132,133,134,
                                136,137,138,
                                140,141,142,
                                144,145,146,
                                148,149,150,
                                152,153,154,
                                156,157,158,
                                160,161,162,
                                164,165,166,
                                168,169,170,
                                172,173,174,
                                176,177,178]], axis=1, inplace=True)
# 归一化
lstm_alone_scaler = MinMaxScaler(feature_range=(0, 1))
lstm_alone_scaled = lstm_alone_scaler.fit_transform(lstm_alone_re_frame)
n_train_days = LSTM_train_days    # 用800天的数据做训练数据
# 划分训练数据与测试数据
lstm_alone_train = lstm_alone_scaled[:n_train_days, :]
lstm_alone_test = lstm_alone_scaled[n_train_days:, :]
# 拆分输入输出
lstm_alone_train_X, lstm_alone_train_y = lstm_alone_train[:, :-15], lstm_alone_train[:, -15:]
lstm_alone_test_X, lstm_alone_test_y = lstm_alone_test[:, :-15], lstm_alone_test[:, -15:]
# reshape input to be 3D [samples, timesteps, features]
lstm_alone_train_X = lstm_alone_train_X.reshape((lstm_alone_train_X.shape[0], 1, lstm_alone_train_X.shape[1]))
lstm_alone_test_X = lstm_alone_test_X.reshape((lstm_alone_test_X.shape[0], 1, lstm_alone_test_X.shape[1]))
# design network
lstm_alone_model = Sequential()
lstm_alone_model.add(LSTM(120, input_shape=(lstm_alone_train_X.shape[1], lstm_alone_train_X.shape[2])))
lstm_alone_model.add(Dense(80))
lstm_alone_model.add(Dense(15))
lstm_alone_model.compile(loss='mae', optimizer='adam')
# fit network
lstm_alone_history = lstm_alone_model.fit(lstm_alone_train_X, lstm_alone_train_y, epochs=1000, batch_size=45, validation_data=(lstm_alone_test_X, lstm_alone_test_y), verbose=2, shuffle=False)
# plot history
# plt.plot(lstm_alone_history.history['loss'], label='train')
# plt.plot(lstm_alone_history.history['val_loss'], label='test')
# plt.legend()
# plt.show()
# make a prediction
lstm_alone_yhat = lstm_alone_model.predict(lstm_alone_test_X)    # 预测是用预测集的前30天预测后15天，滚动预测
lstm_alone_test_X = lstm_alone_test_X.reshape((lstm_alone_test_X.shape[0], lstm_alone_test_X.shape[2]))
# invert scaling for forecast
lstm_alone_inv_yhat = concatenate((lstm_alone_test_X, lstm_alone_yhat), axis=1)
lstm_alone_inv_yhat = lstm_alone_scaler.inverse_transform(lstm_alone_inv_yhat)
lstm_alone_inv_yhat = lstm_alone_inv_yhat[:,-15:]
# 转置
lstm_alone_inv_yhat_transpose = pd.DataFrame([i for i in range(lstm_alone_inv_yhat.shape[1])])
for i in range(lstm_alone_inv_yhat.shape[0]):
    lstm_alone_inv_yhat_transpose = pd.concat((lstm_alone_inv_yhat_transpose, pd.DataFrame(list(lstm_alone_inv_yhat[i]))), axis=1)
lstm_alone_inv_yhat_transpose.columns = [i for i in range(-1, lstm_alone_inv_yhat_transpose.shape[1]-1, 1)]
lstm_alone_inv_yhat_transpose.drop(lstm_alone_inv_yhat_transpose.columns[[0]], axis=1, inplace=True)

# invert scaling for actual
lstm_alone_inv_y = concatenate((lstm_alone_test_X, lstm_alone_test_y), axis=1)
lstm_alone_inv_y = lstm_alone_scaler.inverse_transform(lstm_alone_inv_y)
lstm_alone_inv_y = lstm_alone_inv_y[:,-15:]
# 转置
lstm_alone_inv_y_transpose = pd.DataFrame([i for i in range(lstm_alone_inv_y.shape[1])])
for i in range(lstm_alone_inv_y.shape[0]):
    lstm_alone_inv_y_transpose = pd.concat((lstm_alone_inv_y_transpose, pd.DataFrame(list(lstm_alone_inv_y[i]))), axis=1)
lstm_alone_inv_y_transpose.columns = [i for i in range(-1, lstm_alone_inv_y_transpose.shape[1]-1, 1)]
lstm_alone_inv_y_transpose.drop(lstm_alone_inv_y_transpose.columns[[0]], axis=1, inplace=True)
############################################## 单独的LSTM神经网络结束 ####################################################
# 三预测结果整合
start_rows = 851
target_df = pd.DataFrame(columns=('date', 'primary', 'BP', 'LSTM', 'BP_LSTM', 'BP_err_rate', 'LSTM_err_rate', 'BP_LSTM_err_rate'))
target_df.date = df.date_time.loc[start_rows:start_rows+LSTM_pred_days-1].reset_index(drop=True)
target_df.primary = df.gas_use.loc[start_rows:start_rows+LSTM_pred_days-1].reset_index(drop=True)
target_df.BP= result_hg.y_hg.loc[start_rows:start_rows+LSTM_pred_days-1].reset_index(drop=True)
target_df.BP_err_rate = result_hg.hg_err_rate.loc[start_rows:start_rows+LSTM_pred_days-1].reset_index(drop=True)
target_df.BP_LSTM = BP_LSTM_result.loc[:, [start_rows-LSTM_test_days-LSTM_learn_days]]
target_df.BP_LSTM_err_rate = (target_df.BP_LSTM-target_df.primary)/target_df.primary
target_df.LSTM = lstm_alone_inv_yhat_transpose.loc[:, 0]
target_df.LSTM_err_rate = (target_df.LSTM-target_df.primary)/target_df.primary

print('BP 平均误差为：', abs(target_df.BP_err_rate).mean()*100, '%')
print('LSTM 平均误差为：', abs(target_df.LSTM_err_rate).mean()*100, '%')
print('BP_LSTM 平均误差为：', abs(target_df.BP_LSTM_err_rate).mean()*100, '%')
# 三预测结果图绘图
mpl.rcParams['font.sans-serif'] = ['SimHei']
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(target_df.date, target_df.primary, '-', lw=2, label=u'Primary')
ax1.plot(target_df.date, target_df.BP, 'g--*', lw=2, label=u'BPNN')
ax1.plot(target_df.date, target_df.LSTM, 'c-.^', lw=2, label=u'LSTM')
ax1.plot(target_df.date, target_df.BP_LSTM, 'm:o', lw=2, label=u'BP_LSTM')
ht_holiday = pd.DataFrame(df.is_holiday.loc[start_rows:start_rows+14]).reset_index(drop=True)
# ax1.scatter(target_df.date.values, 500000*ht_holiday.is_holiday.values, c='r', marker='^', label=u'holiday')
plt.title(u"(a) 预测结果比较")

ax2 = fig.add_subplot(212)
ax2.plot(target_df.date, target_df.BP_err_rate, 'g--*', lw=2, label=u'BPNN')
ax2.plot(target_df.date, target_df.LSTM_err_rate, 'c-.^', lw=2, label=u'LSTM')
ax2.plot(target_df.date, target_df.BP_LSTM_err_rate, 'm:o', lw=2, label=u'BP_LSTM')
plt.title(u"(b) 误差率")

ax1.legend(loc='upper right')
ax2.legend(loc='upper right')

plt.show()

# 均方根误差计算
def RMSE_cal(primary_list, pred_list):
    err_list = []
    for i in range(len(primary_list)):
        err_list.append((primary_list[i]-pred_list[i])*(primary_list[i]-pred_list[i]))
    return math.sqrt(sum(err_list)/len(primary_list))
print('RMSE---BP:', RMSE_cal(target_df.primary, target_df.BP))
print('RMSE---LSTM:', RMSE_cal(target_df.primary, target_df.LSTM))
print('RMSE---BP-LSTM:', RMSE_cal(target_df.primary, target_df.BP_LSTM))
print('R---BP:', target_df.primary.corr(target_df.BP))
print('R---LSTM:', target_df.primary.corr(target_df.LSTM))
print('R---BP_LSTM:', target_df.primary.corr(target_df.BP_LSTM))