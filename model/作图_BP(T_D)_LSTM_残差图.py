# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 11:38:14 2018

@author: chenc
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import *  
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
LSTM_test_days = 800    # 测试集从此开始
# 设置plt格式
sns.set_style("white")
# 读取数据集导入数据
df = pd.read_excel('.\data\居&商_武汉20150101~20170823.xlsx')
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
hg_err_mean = result_hg.hg_err_rate.mean()

plt.plot(result_hg.ds,result_hg.hg_err)
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

# 寻找平均误差最小的预测值 的起始行数
min_err_row = BP_LSTM_err_rate.index(min(BP_LSTM_err_rate)) + LSTM_test_days + LSTM_learn_days

# 所需数据提取
target_df = pd.DataFrame(columns=('date', 'primary', 'BP', 'BP_LSTM', 'BP_err_rate', 'BP_LSTM_err_rate'))
target_df.date = df.date_time.loc[min_err_row:min_err_row+LSTM_pred_days-1].reset_index(drop=True)
target_df.primary = df.gas_use.loc[min_err_row:min_err_row+LSTM_pred_days-1].reset_index(drop=True)
target_df.BP= result_hg.y_hg.loc[min_err_row:min_err_row+LSTM_pred_days-1].reset_index(drop=True)
target_df.BP_err_rate = result_hg.hg_err_rate.loc[min_err_row:min_err_row+LSTM_pred_days-1].reset_index(drop=True)
target_df.BP_LSTM = BP_LSTM_result.loc[:, [min_err_row-LSTM_test_days-LSTM_learn_days]]
target_df.BP_LSTM_err_rate = (target_df.BP_LSTM-target_df.primary)/target_df.primary
################################################# LSTM神经网络结束 #######################################################

# BP训练结果图与残差图
mpl.rcParams['font.sans-serif'] = ['SimHei']
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(result_hg.ds[0:850], result_hg.y[0:850], '-', lw=1.5, label=u'实际结果')
ax1.plot(result_hg.ds[0:850], result_hg.y_hg[0:850], 'g--', lw=1.5, label=u'预测结果')
plt.title(u"(a) BPNN预测结果")

ax2 = fig.add_subplot(212)
ax2.plot(result_hg.ds[0:850], result_hg.hg_err[0:850], 'r-', lw=1.5, label=u'残差')
#ax2.plot(result_hg.ds[0:850], 100000*result_hg.ave_temp[0:850], 'b', lw=1.5, label=u'日平均气温')
plt.title(u"(b) 残差序列")
ax1.legend(loc='upper right')
ax2.legend(loc='upper right')

plt.show()
print(abs(target_df.BP_LSTM_err_rate).mean())
print(abs(target_df.BP_err_rate).mean())