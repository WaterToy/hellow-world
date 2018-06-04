import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
#   from sklearn.preprocessing import MinMaxScaler
from pylab import *  

# 设置plt格式
sns.set_style("white")  
mpl.rcParams['font.sans-serif'] = ['SimHei']  


# 导入输入、输出值
# 读取数据集导入数据
df = pd.read_excel('.\\data\\gas_load.xlsx')
# 异常数据直接剔除
# df.gas_use[601] = (df.gas_use[600]+df.gas_use[602])/2
# df.gas_use[614] = (df.gas_use[613]+df.gas_use[615])/2
df = df[['date_time', 'is_holiday', 'gas_use','ave_temp']].loc[0:865]


x = df.date_time
# scaler1 = MinMaxScaler(feature_range=(0, 1))
y1 = df.gas_use
y2 = df.ave_temp

fig = plt.figure(figsize=(17,6))
ax1 = fig.add_subplot(111)
ax1.plot(x.values, y1.values, '-', label=u'燃气日负荷')
ax1.scatter(x1.values, y3.values, c='r', '^', label=u'节假日')
ax1.set_ylabel(u'规则化的燃气日负荷')

ax2 = ax1.twinx()  # this is the important function
ax2.plot(x.values, y2.values, '--g',label=u'日平均气温')
ax2.set_ylabel(u'日平均气温')

ax1.legend(loc='lower right')
ax2.legend(loc='0')

plt.show()
