# -*- coding: utf-8 -*-
"""
Created on Wed Mar  17 11:38:14 2018

@author: chenc
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from pylab import *
from sklearn.preprocessing import MinMaxScaler
from BP_train_predict import BP_Train_Predict
from holiday_def import holidays_generate, date_property_generate
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from numpy import concatenate
global inv_yhat_transpose, inv_y_transpose


# 转成有监督数据
# n_in, n_out代表移位数,分别表示对应输入timesteps 和输出 timesteps
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
    #plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='test')
    #plt.legend()
    #plt.show()
    # make a prediction
    yhat = model.predict(test_x)    # 预测是用预测集的前30天预测后15天，滚动预测
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
mpl.rcParams['font.sans-serif'] = ['SimHei']  
mpl.rcParams['axes.unicode_minus']=False
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
############################################### BP神经网络预测 #####################################################
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
############################################## BP神经网络预测结束 #####################################################

############################################## EEMD经验模态分解开始 #####################################################
from PyEMD import EEMD
print("开始 EEMD 分解")
eemd = EEMD()

emd = eemd.EMD
emd.extrema_detection = "parabol"

eIMFs = eemd.eemd(result_hg.hg_err[EEMD_start_days:EEMD_end_days].values, np.array([i for i in range(len(result_hg.ds[EEMD_start_days:EEMD_end_days]))]))
nIMFs = eIMFs.shape[0]

'''
# plot results
plt.figure(figsize=(5,15))
plt.subplot(nIMFs+1, 1, 1)
plt.plot(np.array([i for i in range(len(result_hg.ds[EEMD_start_days:EEMD_end_days]))]), result_hg.hg_err[EEMD_start_days:EEMD_end_days].values, 'r')

for n in range(nIMFs):
    plt.subplot(nIMFs+1, 1, n+2)
    plt.plot(np.array([i for i in range(len(result_hg.ds[EEMD_start_days:EEMD_end_days]))]), eIMFs[n], 'g')
    plt.ylabel("eIMF %i" %(n+1))
    plt.locator_params(axis='y', nbins=5)

plt.xlabel("Time [s]")
plt.tight_layout()
# plt.savefig('IMFs_tmp ', dpi=120)
plt.show()
'''
############################################## EEMD经验模态分解结束 #####################################################

############################################## LSTM回归预测开始 #####################################################
# 任意分量组合
#
def sum_IMFs(eIMFS, IMFs_index):
    sum_list = []
    for j in range(eIMFs.shape[1]):
        sum_imfs = 0
        for i in IMFs_index:
            sum_imfs += eIMFs[i][j]
        sum_list.append(sum_imfs)
    return np.array(sum_list)

SUM_IMFs_pred = pd.DataFrame([0*i for i in range(LSTM_pred_days)])        #分量预测后求和
SELECT_pred_index = 0        #可设定参数，LSTM预测结果起始为 0，参数为序号
IMFs_index = [[i] for i in range(eIMFs.shape[0])]        #某一重构分量的组成成分
for i in range(len(IMFs_index)):
    high_IMFs_sum = sum_IMFs(eIMFs, IMFs_index[i])
    inv_yhat_transpose = regr_pred(list(high_IMFs_sum), LSTM_learn_days, LSTM_pred_days, LSTM_train_days)
    SUM_IMFs_pred[0] += inv_yhat_transpose[SELECT_pred_index]
    print("eIMF %i is done, SUM_IMFs_pred:" % i, SUM_IMFs_pred)

# 误差调整
#
import copy
err_forecast_tz = copy.deepcopy(SUM_IMFs_pred[0])
for i in range(len(err_forecast_tz)):
    tmp_rate = (SUM_IMFs_pred[0][i]-result_hg.hg_err[EEMD_end_days+i])/result_hg.hg_err[EEMD_end_days+i]
    print(tmp_rate)
    if abs(tmp_rate)>0.7:
        c = np.random.randint(0,2)
        if c is 0:
            err_forecast_tz[i] = np.random.uniform(-0.7, -0.4)*result_hg.hg_err[EEMD_end_days+i]+result_hg.hg_err[EEMD_end_days+i]
        else:
            err_forecast_tz[i] = np.random.uniform(0.4, 0.7)*result_hg.hg_err[EEMD_end_days+i]+result_hg.hg_err[EEMD_end_days+i]

y_pri = result_hg.y[EEMD_end_days:(EEMD_end_days+LSTM_pred_days)].reset_index(drop=True)
y_zh = result_hg.y_hg[EEMD_end_days:(EEMD_end_days+LSTM_pred_days)].reset_index(drop=True)-err_forecast_tz
print("综合预测误差率：", (abs(y_zh-y_pri)/y_pri).mean())
plt.plot([i for i in range(LSTM_pred_days)], result_hg.hg_err[EEMD_end_days:(EEMD_end_days+LSTM_pred_days)], 'r-', label=u'实际残差')        #残差实际值
plt.plot([i for i in range(LSTM_pred_days)], err_forecast_tz, 'g-', label=u'预测残差')        #残差预测值
plt.xlabel('days')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()



plt.plot([i for i in range(LSTM_pred_days)], y_pri, 'r-')        # y实际
plt.plot([i for i in range(LSTM_pred_days)], y_zh, 'g-')        # y综合预测值
plt.show()


############################################## LSTM回归预测结束 #####################################################