import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pylab import *  
import matplotlib.dates as mdate

# 设置plt格式
sns.set_style("white")  
mpl.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
mpl.rcParams['axes.unicode_minus']=False #用来正常显示负号


# 导入输入、输出值
# 读取数据集导入数据
df = pd.read_excel('D:\\Mission\\云同步\\学习\\20180124_小论文\\20171123数据整理\武汉\居&商_武汉20150101~20170823.xlsx')
df = df.loc[0:850]
x = df.date_time
y1 = df.gas_use
y2 = df.is_holiday
y3 = df.date_time.loc[y2==2].reset_index(drop=True)
y4 = df.is_holiday.loc[y2==2].reset_index(drop=True)

fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.scatter(y3, 4800*y4.values, c='orangered', marker='^', label=u'BPNN_EMD_LSTM')
plt.plot(x, y1, c='darkcyan')
ax1.set_ylabel(u'燃气日负荷')
ax1.legend(loc='upper right')
plt.show()
