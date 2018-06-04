import pandas as pd
import numpy as np
import pymysql
from fbprophet import Prophet
# from BPNetwork import BPNeuralNetwork as bpnn
import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn import linear_model
# from sklearn.preprocessing import PolynomialFeatures

# sns.set_style()
'''
# 回归预测部分
# 读取数据集导入数据
df = pd.read_excel('.\\data\\gas_load.xlsx')
df_not_holiday = df.loc[df['is_holiday']==0]
datasets_temp = df_not_holiday.ave_temp
datasets_gas_use = df_not_holiday.gas_use
test_forecast = df_not_holiday.ave_temp

length = len(datasets_temp)
datasets_temp = np.array(datasets_temp).reshape([length,1])
datasets_gas_use = np.array(datasets_gas_use)


min_temp = min(datasets_temp)
max_temp = max(datasets_temp)
X = np.arange(min_temp,max_temp).reshape([-1,1])
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(datasets_temp)
lin_reg_2 = linear_model.LinearRegression()
lin_reg_2.fit(X_poly, datasets_gas_use)

predict_outcome = lin_reg_2.predict(X_poly)
predictions = {}
predictions['intercept'] = lin_reg_2.intercept_
predictions['coefficient'] = lin_reg_2.coef_
predictions['predicted_value'] = predict_outcome
'''

# fbprophet 部分
df = pd.read_excel('H:\\20171123数据整理\武汉\居&商_武汉20150101~20170823.xlsx')
conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='WHUTieb10-8.', db='gas_forecast', charset='utf8')
cur = conn.cursor()
cur.execute("SELECT date, is_holiday FROM holiday WHERE date>='%s' AND date<='%s' AND is_holiday=2" % ('2014-01-01', '2018-01-01'))
holiday = cur.fetchall()
list_holiday = []
forecast_result = pd.DataFrame(columns=('ds', 'y', 'y_fb', 'err', 'err_rate', 'is_holiday', 'ave_temp'))
for i in range(len(holiday)):
    list_holiday.append(holiday[i][0])
holidays = pd.DataFrame({
    'holiday': '法定节假日',
    'ds': list_holiday,
    'lower_window': 0,
    'upper_window': 0,
                         })

'''
# 一次预测7天，循环预测验证30轮
for i in range(30):
    new = pd.DataFrame(columns=('ds', 'y'))
    tmp = df[0:731+7*i]
    new.ds = tmp.date_time
    new.y = np.log(tmp.gas_use)
    prophet = Prophet(holidays=holidays)
    prophet.fit(new)
    future = prophet.make_future_dataframe(freq='D', periods=7)
    forecasts = prophet.predict(future)
    # forecasts[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    # prophet.plot(forecasts)
    # prophet.plot_components(forecasts)
    
    # 数据还原
    tmp1 = forecasts.tail(7)
    for j in range(7):
        tmp2 = np.exp(tmp1.yhat)
        ds = df.date_time[731+7*i+j]
        y = df.gas_use[731+7*i+j]
        y_fb = float(tmp2[j:j+1].values)
        err = y - y_fb
        err_rate = abs(err)/y
        is_holiday = df.is_holiday[731+7*i+j]
        ave_temp = df.ave_temp[731+7*i+j]
        forecast_result.loc[i*7+j] = {'ds': ds, 'y': y, 'y_fb': y_fb, 'err': err, 'err_rate': err_rate, 'is_holiday': is_holiday, 'ave_temp': ave_temp}
'''

new = pd.DataFrame(columns=('ds', 'y'))
tmp = df[0:731]
new.ds = tmp.date_time
new.y = np.log(tmp.gas_use)
prophet = Prophet(growth='logistic', holidays=holidays)
new['cap'] = 15.61
prophet.fit(new)
future = prophet.make_future_dataframe(freq='D', periods=30)
future['cap'] = 15.61
forecasts = prophet.predict(future)
forecasts[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
prophet.plot(forecasts)
prophet.plot_components(forecasts)

'''
# 一次预测30天，循环预测验证7轮
for i in range(7):
    new = pd.DataFrame(columns=('ds', 'y'))
    tmp = df[0:731+30*i]
    new.ds = tmp.date_time
    new.y = np.log(tmp.gas_use)
    prophet = Prophet(holidays=holidays)
    prophet.fit(new)
    future = prophet.make_future_dataframe(freq='D', periods=30)
    forecasts = prophet.predict(future)
    # forecasts[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    # prophet.plot(forecasts)
    # prophet.plot_components(forecasts)
    
    # 数据还原
    tmp1 = forecasts.tail(30)
    for j in range(30):
        tmp2 = np.exp(tmp1.yhat)
        ds = df.date_time[731+30*i+j]
        y = df.gas_use[731+30*i+j]
        y_fb = float(tmp2[j:j+1].values)
        err = y - y_fb
        err_rate = abs(err)/y
        is_holiday = df.is_holiday[731+30*i+j]
        ave_temp = df.ave_temp[731+30*i+j]
        forecast_result.loc[i*30+j] = {'ds': ds, 'y': y, 'y_fb': y_fb, 'err': err, 'err_rate': err_rate, 'is_holiday': is_holiday, 'ave_temp': ave_temp}
'''