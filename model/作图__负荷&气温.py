import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from pylab import *  

# 设置plt格式
sns.set_style("white")  
mpl.rcParams['font.sans-serif'] = ['SimHei']  


# 导入输入、输出值
# 读取数据集导入数据
df = pd.read_excel('D:\\20180124_小论文\\20171123数据整理\武汉\居&商_武汉20150101~20170823.xlsx')
# 异常数据直接剔除
df.gas_use[601] = (df.gas_use[600]+df.gas_use[602])/2
df.gas_use[614] = (df.gas_use[613]+df.gas_use[615])/2
df = df[['date_time', 'is_holiday', 'gas_use','ave_temp']].loc[0:866]


x = df.date_time
scaler1 = MinMaxScaler(feature_range=(0, 1))
y1 = scaler1.fit_transform(df[['is_holiday', 'gas_use']])
y1 = y1[:,1]
y2 = df.ave_temp

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, '-', label=u'燃气日负荷')
ax1.set_ylabel(u'规则化的燃气日负荷')

ax2 = ax1.twinx()  # this is the important function
ax2.plot(x, y2, '--g',label=u'日平均气温')
ax2.set_ylabel(u'日平均气温')

ax1.legend(loc='lower right')
ax2.legend(loc='0')

plt.show()
