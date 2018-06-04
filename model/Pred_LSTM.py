import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from numpy import concatenate

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

# 读取数据集导入数据
df = pd.read_excel('.\\data\\gas_load.xlsx')
df = dataframe[['ave_temp', 'is_holiday', 'gas_use']]
# 异常数据直接剔除
df.gas_use[601] = (df.gas_use[600]+df.gas_use[602])/2
df.gas_use[614] = (df.gas_use[613]+df.gas_use[615])/2
# 补充日期序号
day_index= [i%365 for i in range(len(df.gas_use))]
df.insert(0, 'day_index', day_index)
df_values = df.values.astype('float32')
# 转化为有监督数据
re_frame = series_to_supervised(df_values, 30, 15)
# 删除不预测的列
re_frame.drop(re_frame.columns[[120,121,122,
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
print(re_frame.head())
# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
reframed = scaler.fit_transform(re_frame)
n_train_days = 724    # 用800天的数据做训练数据
# df_values = reframed.values
# 划分训练数据与测试数据
train = reframed[:n_train_days, :]
test = reframed[n_train_days:, :] 
# 拆分输入输出
train_X, train_y = train[:, :-15], train[:, -15:]
test_X, test_y = test[:, :-15], test[:, -15:]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(15))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=100, batch_size=50, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
# make a prediction
yhat = model.predict(test_X)    # 预测是用预测集的前30天预测后15天，滚动预测
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((test_X, yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,-15:]
# 转置
inv_yhat_transpose = pd.DataFrame([i for i in range(inv_yhat.shape[1])])
for i in range(inv_yhat.shape[0]):
	inv_yhat_transpose = pd.concat((inv_yhat_transpose, pd.DataFrame(list(inv_yhat[i]))), axis=1)
inv_yhat_transpose.columns = [i for i in range(-1, inv_yhat_transpose.shape[1]-1, 1)]
inv_yhat_transpose.drop(inv_yhat_transpose.columns[[0]], axis=1, inplace=True)

# invert scaling for actual
inv_y = concatenate((test_X, test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,-15:]
# 转置
inv_y_transpose = pd.DataFrame([i for i in range(inv_y.shape[1])])
for i in range(inv_y.shape[0]):
	inv_y_transpose = pd.concat((inv_y_transpose, pd.DataFrame(list(inv_y[i]))), axis=1)
inv_y_transpose.columns = [i for i in range(-1, inv_y_transpose.shape[1]-1, 1)]
inv_y_transpose.drop(inv_y_transpose.columns[[0]], axis=1, inplace=True)
# 寻找误差较小的一部分
row_num = inv_y_transpose.shape[1]
err_rate = []
err_rate.append([((abs(inv_yhat_transpose[i]-inv_y_transpose[i])/inv_y_transpose[i])).mean() for i in range(row_num)])
err_rate = err_rate[0]
print('滚动预测误差率分别为：', err_rate)
