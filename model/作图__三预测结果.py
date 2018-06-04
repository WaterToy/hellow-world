import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import holiday_def
from pylab import *  
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
y_data_max, y_data_min = df_no_holiday.gas_use.max(), df_no_holiday.gas_use.min()
y_data = 0.7*(df_no_holiday.gas_use-y_data_min)/(y_data_max-y_data_min)+0.15

np.random.seed(0)
# 构建一个顺序模型
model = Sequential()
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

# 数据还原
result_hg = pd.DataFrame(columns=('ds', 'is_holiday', 'ave_temp', 'y', 'y_hg', 'err_hg', 'err_rate_hg', 'tmp_err_hg'))
result_hg.ds = df.date_time
result_hg.is_holiday = df.is_holiday
result_hg.ave_temp = df.ave_temp
result_hg.y = df.gas_use
result_hg.y_hg = (y_pred_all[:, 0] - 0.15)/0.7 * (y_data_max-y_data_min) + y_data_min
result_hg.err_hg = result_hg.y_hg - result_hg.y    # 预测-实际
result_hg.err_rate_hg = abs(result_hg.err_hg / result_hg.y)
result_hg.tmp_err_hg = result_hg.err_hg / 500000
hg_err_mean = result_hg.err_rate_hg.mean()


# prophet预测部分
result_fb = pd.DataFrame(columns=('ds', 'err_hg_input', 'err_hg_fb_output', 'err_fb', 'err_rate_fb', 'is_holiday', 'ave_temp', 'y', 'y_zh', 'err_zh', 'err_rate_zh'))
# 节假日生成
holidays = holiday_def.holidays_generate()

# 一次预测30天，循环预测验证7轮
for i in range(1):
    new_method1 = pd.DataFrame(columns=('ds', 'y'))
    tmp_method1 = result_hg[0:752+15*i]
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
        ds = result_hg.ds[752+15*i+j]
        err_hg_input = result_hg.err_hg[752+15*i+j]
        # if tmp2_method1.yhat[752+15*i+j]>0:
           # err_hg_fb_output = np.exp(tmp2_method1.yhat[752+15*i+j])
        # else:
           # err_hg_fb_output = -np.exp(-tmp2_method1.yhat[752+15*i+j])
        err_hg_fb_output = tmp2_method1.yhat[752+15*i+j] * 500000
        err_fb = err_hg_fb_output - err_hg_input
        err_rate_fb = abs(err_fb/err_hg_input)
        is_holiday = result_hg.is_holiday[752+15*i+j]
        ave_temp = result_hg.ave_temp[752+15*i+j]
        y = result_hg.y[752+15*i+j]
        y_zh = result_hg.y_hg[752+15*i+j] - err_hg_fb_output
        err_zh = y_zh - y
        err_rate_zh = abs(err_zh/y)
        result_fb.loc[i*15+j] = {'ds': ds, 'err_hg_input': err_hg_input, 'err_hg_fb_output': err_hg_fb_output, 'err_fb': err_fb, 'err_rate_fb': err_rate_fb, 'is_holiday': is_holiday, 'ave_temp': ave_temp, 'y': y, 'y_zh': y_zh, 'err_zh': err_zh, 'err_rate_zh':err_rate_zh}

##########################################################################################################
#                                               fb开源工具预测结果
##########################################################################################################
forecast_result2 = pd.DataFrame(columns=('ds', 'y', 'y_fb', 'err', 'err_rate', 'is_holiday', 'ave_temp'))
for i in range(1):
    new_method2 = pd.DataFrame(columns=('ds', 'y'))
    tmp_method2 = df[0:752+15*i]
    new_method2.ds = tmp_method2.date_time
    new_method2.y = np.log(tmp_method2.gas_use)
    prophet2 = Prophet(holidays=holidays)
    prophet2.fit(new_method2)
    future2 = prophet2.make_future_dataframe(freq='D', periods=15)
    forecasts2 = prophet2.predict(future2)
    forecasts[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    prophet.plot(forecasts)
    prophet.plot_components(forecasts)
    
    # 数据还原
    tmp1_method2 = forecasts2.tail(15)
    for j in range(15):
        tmp2_method2 = np.exp(tmp1_method2.yhat)
        ds = df.date_time[752+15*i+j]
        y = df.gas_use[752+15*i+j]
        y_fb = float(tmp2_method2[j:j+1].values)
        err = y_fb - y
        err_rate = abs(err)/y
        is_holiday = df.is_holiday[752+15*i+j]
        ave_temp = df.ave_temp[752+15*i+j]
        forecast_result2.loc[i*15+j] = {'ds': ds, 'y': y, 'y_fb': y_fb, 'err': err, 'err_rate': err_rate, 'is_holiday': is_holiday, 'ave_temp': ave_temp}

# 新加入的
result_fb2 = pd.DataFrame(columns=('ds', 'err_hg_input', 'err_hg_fb_output', 'err_fb', 'err_rate_fb', 'is_holiday', 'ave_temp', 'y', 'y_zh', 'err_zh', 'err_rate_zh'))
for i in range(1):
    new_method3 = pd.DataFrame(columns=('ds', 'y'))
    tmp_method3 = result_hg[0:731+15*i]
    new_method3.ds = tmp_method3.ds
    new_method3.y = tmp_method3.tmp_err_hg
    prophet3 = Prophet(holidays=holidays)
    prophet3.fit(new_method3)
    future = prophet3.make_future_dataframe(freq='D', periods=15)
    forecasts3 = prophet3.predict(future)
    forecasts3[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    prophet3.plot(forecasts3)
    prophet3.plot_components(forecasts3)
    
    # 数据还原
    tmp2_method3 = forecasts3.tail(15)
    for j in range(15):
        ds = result_hg.ds[731+15*i+j]
        err_hg_input = result_hg.err_hg[731+15*i+j]
        err_hg_fb_output = tmp2_method3.yhat[731+15*i+j]*500000
        err_fb = err_hg_fb_output - err_hg_input
        err_rate_fb = abs(err_fb/err_hg_input)
        is_holiday = result_hg.is_holiday[731+15*i+j]
        ave_temp = result_hg.ave_temp[731+15*i+j]
        y = result_hg.y[731+15*i+j]
        y_zh = result_hg.y_hg[731+15*i+j] - err_hg_fb_output
        err_zh = y_zh - y
        err_rate_zh = abs(err_zh/y)
        result_fb2.loc[i*15+j] = {'ds': ds, 'err_hg_input': err_hg_input, 'err_hg_fb_output': err_hg_fb_output, 'err_fb': err_fb, 'err_rate_fb': err_rate_fb, 'is_holiday': is_holiday, 'ave_temp': ave_temp, 'y': y, 'y_zh': y_zh, 'err_zh': err_zh, 'err_rate_zh':err_rate_zh}

##########################################################################################################
#                                               fb开源工具预测结果
##########################################################################################################
forecast_result3 = pd.DataFrame(columns=('ds', 'y', 'y_fb', 'err', 'err_rate', 'is_holiday', 'ave_temp'))
for i in range(1):
    new_method2 = pd.DataFrame(columns=('ds', 'y'))
    tmp_method2 = df[0:731+15*i]
    new_method2.ds = tmp_method2.date_time
    new_method2.y = np.log(tmp_method2.gas_use)
    prophet4 = Prophet(holidays=holidays)
    prophet4.fit(new_method2)
    future4 = prophet4.make_future_dataframe(freq='D', periods=15)
    forecasts4 = prophet4.predict(future4)
    # forecasts[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    # prophet.plot(forecasts)
    # prophet.plot_components(forecasts)
    
    # 数据还原
    tmp1_method2 = forecasts4.tail(15)
    for j in range(15):
        tmp2_method2 = np.exp(tmp1_method2.yhat)
        ds = df.date_time[731+15*i+j]
        y = df.gas_use[731+15*i+j]
        y_fb = float(tmp2_method2[j:j+1].values)
        err = y_fb - y
        err_rate = abs(err)/y
        is_holiday = df.is_holiday[731+15*i+j]
        ave_temp = df.ave_temp[731+15*i+j]
        forecast_result3.loc[i*15+j] = {'ds': ds, 'y': y, 'y_fb': y_fb, 'err': err, 'err_rate': err_rate, 'is_holiday': is_holiday, 'ave_temp': ave_temp}
# 设置plt格式
sns.set_style("white")  
mpl.rcParams['font.sans-serif'] = ['SimHei']  

x1 = df.date_time.loc[752:766]
x2 = df.date_time.loc[731:745]
y1 = df.gas_use.loc[752:766]
y2 = df.gas_use.loc[731:745]
y_BP1 = result_hg.y_hg.loc[752:766]
y_BP2 = result_hg.y_hg.loc[731:745]
y_GAM1 = forecast_result2.y_fb
y_GAM2 = forecast_result3.y_fb
y_ZH1 = result_fb.y_zh
y_ZH2 = result_fb2.y_zh


fig = plt.figure()

ax1 = fig.add_subplot(223)
# ax1.scatter(y4.index, y4*6.75, c='r', marker='*', label=u'节假日')
ax1.plot(x1, y1, '-', label=u'实际值')
ax1.plot(x1, y_BP1, 'y--*', label=u'BP')
ax1.plot(x1, y_GAM1, 'g--^', label=u'GAM')
ax1.plot(x1, y_ZH1, 'r--o', label=u'BP-GAM')
ax1.legend(loc='lower right')

ax2 = fig.add_subplot(221)
# ax1.scatter(y4.index, y4*6.75, c='r', marker='*', label=u'节假日')
ax2.plot(x2, y2, '-', label=u'实际值')
ax2.plot(x2, y_BP2, 'y--*', label=u'BP')
ax2.plot(x2, y_GAM2, 'g--^', label=u'GAM')
ax2.plot(x2, y_ZH2, 'r--o', label=u'BP-GAM')
ax2.legend(loc='lower right')

y = df.gas_use.loc[752:766].reset_index(drop=True)
y2 = df.gas_use.loc[731:745].reset_index(drop=True)
y_BP1 = result_hg.y_hg.loc[752:766].reset_index(drop=True)
y_BP2 = result_hg.y_hg.loc[731:745].reset_index(drop=True)
y_GAM1 = forecast_result2.y_fb
y_GAM2 = forecast_result3.y_fb
y_ZH1 = result_fb.y_zh
y_ZH2 = result_fb2.y_zh

ax3 = fig.add_subplot(224)
# ax1.scatter(y4.index, y4*6.75, c='r', marker='*', label=u'节假日')
ax3.plot((y_BP1-y)/y, 'y--*', label=u'BP')
ax3.plot((y_GAM1-y)/y, 'g--^', label=u'GAM')
ax3.plot((y_ZH1-y)/y, 'r--o', label=u'BP-GAM')
ax3.legend(loc='upper right')

ax4 = fig.add_subplot(222)
# ax1.scatter(y4.index, y4*6.75, c='r', marker='*', label=u'节假日')
ax4.plot((y_BP2-y2)/y2, 'y--*', label=u'BP')
ax4.plot((y_GAM2-y2)/y2, 'g--^', label=u'GAM')
ax4.plot((y_ZH2-y2)/y2, 'r--o', label=u'BP-GAM')
ax4.legend(loc='upper right')

plt.show()

# 结果分析

y_BP_mean1 = abs((y_BP1-y)/y).mean()
y_GAM_mean1 = abs((y_GAM1-y)/y).mean()
y_ZH_mean1 = abs((y_ZH1-y)/y).mean()
print('y_BP_mean1:%4f\ny_GAM_mean1:%4f\ny_ZH_mean1:%4f'%(y_BP_mean1,y_GAM_mean1,y_ZH_mean1))

y_BP_mean2 = abs((y_BP2-y)/y).mean()
y_GAM_mean2 = abs((y_GAM2-y)/y).mean()
y_ZH_mean2 = abs((y_ZH2-y)/y).mean()
print('y_BP_mean2:%4f\ny_GAM_mean2:%4f\ny_ZH_mean2:%4f'%(y_BP_mean2,y_GAM_mean2,y_ZH_mean2))
