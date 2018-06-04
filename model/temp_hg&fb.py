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
df = pd.read_excel('D:\\20180124_小论文\\20171123数据整理\武汉\居&商_武汉20150101~20170823.xlsx')
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

plt.scatter(df.ave_temp, 0.7*(df.gas_use-y_data_min)/(y_data_max-y_data_min)+0.15)

plt.plot(df.ave_temp, y_pred_all, 'r-', lw=3)
plt.show()


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



# prophet预测部分
conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='WHUTieb10-8.', db='gas_forecast', charset='utf8')
cur = conn.cursor()
# cur.execute("SELECT date, is_holiday FROM holiday WHERE date>='%s' AND date<='%s' AND is_holiday=2" % ('2014-01-01', '2018-01-01'))
# holiday = cur.fetchall()
# list_holiday = []
result_fb = pd.DataFrame(columns=('ds', 'err_hg_input', 'err_hg_fb_output', 'err_fb', 'err_rate_fb', 'is_holiday', 'ave_temp', 'y', 'y_zh', 'err_zh', 'err_rate_zh'))
# for i in range(len(holiday)):
#     list_holiday.append(holiday[i][0])
holiday_1 = pd.DataFrame({
    'holiday': '2015春节',
    'ds': ['2015-02-19'],
    'lower_window': -5,
    'upper_window': 5,
                         })
holiday_2 = pd.DataFrame({
    'holiday': '2015中秋',
    'ds': ['2015-09-27'],
    'lower_window': -1,
    'upper_window': 0,
                         })
holiday_3 = pd.DataFrame({
    'holiday': '2015国庆',
    'ds': ['2015-10-03'],
    'lower_window': -2,
    'upper_window': 4,
                         })
holiday_4 = pd.DataFrame({
    'holiday': '2016春节',
    'ds': ['2016-02-10'],
    'lower_window': -9,
    'upper_window': 5,
                         })
holiday_5 = pd.DataFrame({
    'holiday': '2016国庆',
    'ds': ['2016-10-03'],
    'lower_window': -4,
    'upper_window': 4,
                         })
holiday_6 = pd.DataFrame({
    'holiday': '2017春节',
    'ds': ['2016-01-28'],
    'lower_window': -7,
    'upper_window': 5,
                         })
holiday_7 = pd.DataFrame({
    'holiday': '3天节假日',
    'ds': ['2015-01-02', '2015-04-05', '2015-05-02', '2015-06-21', '2015-09-04', '2016-01-02', '2016-04-03', '2016-05-01', '2016-06-10', '2016-09-16', '2017-01-01', '2017-04-30', '2017-05-29'],
    'lower_window': -1,
    'upper_window': 1,
                         })
holidays = pd.concat((holiday_1, holiday_2, holiday_3, holiday_4, holiday_5, holiday_6, holiday_7))


# 一次预测30天，循环预测验证7轮
for i in range(10):
    new_method1 = pd.DataFrame(columns=('ds', 'y'))
    tmp_method1 = result_hg[0:800+15*i]
    new_method1.ds = tmp_method1.ds
    new_method1.y = tmp_method1.tmp_err_hg
    prophet = Prophet(holidays=holidays)
    prophet.fit(new_method1)
    future = prophet.make_future_dataframe(freq='D', periods=15)
    forecasts = prophet.predict(future)
    # forecasts[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    # prophet.plot(forecasts)
    # prophet.plot_components(forecasts)
    
    # 数据还原
    tmp2_method1 = forecasts.tail(15)
    for j in range(15):
        ds = result_hg.ds[800+15*i+j]
        err_hg_input = result_hg.err_hg[800+15*i+j]
        err_hg_fb_output = tmp2_method1.yhat[800+15*i+j]*500000
        err_fb = err_hg_fb_output - err_hg_input
        err_rate_fb = abs(err_fb/err_hg_input)
        is_holiday = result_hg.is_holiday[800+15*i+j]
        ave_temp = result_hg.ave_temp[800+15*i+j]
        y = result_hg.y[800+15*i+j]
        y_zh = result_hg.y_hg[800+15*i+j] - err_hg_fb_output
        err_zh = y_zh - y
        err_rate_zh = abs(err_zh/y)
        result_fb.loc[i*15+j] = {'ds': ds, 'err_hg_input': err_hg_input, 'err_hg_fb_output': err_hg_fb_output, 'err_fb': err_fb, 'err_rate_fb': err_rate_fb, 'is_holiday': is_holiday, 'ave_temp': ave_temp, 'y': y, 'y_zh': y_zh, 'err_zh': err_zh, 'err_rate_zh':err_rate_zh}

##########################################################################################################
#                                               fb开源工具预测结果
##########################################################################################################
forecast_result2 = pd.DataFrame(columns=('ds', 'y', 'y_fb', 'err', 'err_rate', 'is_holiday', 'ave_temp'))
for i in range(10):
    new_method2 = pd.DataFrame(columns=('ds', 'y'))
    tmp_method2 = df[0:800+15*i]
    new_method2.ds = tmp_method2.date_time
    new_method2.y = np.log(tmp_method2.gas_use)
    prophet = Prophet(holidays=holidays)
    prophet.fit(new_method2)
    future = prophet.make_future_dataframe(freq='D', periods=15)
    forecasts = prophet.predict(future)
    # forecasts[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    # prophet.plot(forecasts)
    # prophet.plot_components(forecasts)
    
    # 数据还原
    tmp1_method2 = forecasts.tail(15)
    for j in range(15):
        tmp2_method2 = np.exp(tmp1_method2.yhat)
        ds = df.date_time[800+15*i+j]
        y = df.gas_use[800+15*i+j]
        y_fb = float(tmp2_method2[j:j+1].values)
        err = y_fb - y
        err_rate = abs(err)/y
        is_holiday = df.is_holiday[800+15*i+j]
        ave_temp = df.ave_temp[800+15*i+j]
        forecast_result2.loc[i*15+j] = {'ds': ds, 'y': y, 'y_fb': y_fb, 'err': err, 'err_rate': err_rate, 'is_holiday': is_holiday, 'ave_temp': ave_temp}

# 结果分析
y = df.gas_use.loc[800:949].reset_index(drop=True)
y_BP = result_hg.y_hg.loc[800:949].reset_index(drop=True)
y_GAM = forecast_result2.y_fb
y_ZH = result_fb.y_zh
fig = plt.figure()

ax1 = fig.add_subplot(211)
# ax1.scatter(y4.index, y4*6.75, c='r', marker='*', label=u'节假日')
ax1.plot((y_BP-y)/y, '--*', label=u'BP')
ax1.plot((y_GAM-y)/y, '--^', label=u'GAM')
ax1.plot((y_ZH-y)/y, '--o', label=u'BP-GAM')
ax1.legend(loc='best')

ax2 = fig.add_subplot(212)

ax2.plot(y,'r')
ax2.plot(y_BP,'g')
ax2.plot(y_GAM,'b')
ax2.plot(y_ZH,'--')

plt.show()