import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pylab import *  

# 设置plt格式
sns.set_style("white")  
mpl.rcParams['font.sans-serif'] = ['SimHei']  


# 导入输入、输出值
# 读取数据集导入数据
df = pd.read_excel('.\\data\\gas_load.xlsx')

x = df.date_time
y1 = df.gas_use
y3 = df.is_holiday
y4 = y3.loc[y3==2]

fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.scatter(y4.index, 4800*y4, c='r', marker='^', label=u'节假日')
ax1.plot(x.index, y1, '-')
ax1.set_ylabel(u'燃气日负荷')
#ax1.set_title(u"燃气日负荷趋势图")

ax1.legend(loc='0')

plt.show()
