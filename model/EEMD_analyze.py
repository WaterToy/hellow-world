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

# 设置plt格式
sns.set_style("white")
# 读取数据集导入数据
df = pd.read_excel('D:\\Mission\\云同步\\学习\\20180124_小论文\\20171123数据整理\武汉\居&商_武汉20150101~20170823.xlsx')
# 异常数据直接剔除
df.gas_use[601] = (df.gas_use[600]+df.gas_use[602])/2
df.gas_use[614] = (df.gas_use[613]+df.gas_use[615])/2

from PyEMD import EEMD

eemd = EEMD()

emd = eemd.EMD
emd.extrema_detection = "parabol"

eIMFs = eemd.eemd(df.gas_use[0:850].values, np.array([i for i in range(850)]))
nIMFs = eIMFs.shape[0]

# plot results
plt.figure(figsize=(12,30))
plt.subplot(nIMFs+1, 1, 1)
plt.plot(np.array([i for i in range(850)]), df.gas_use[0:850].values, 'r')



for n in range(nIMFs):
    plt.subplot(nIMFs+1, 1, n+2)
    plt.plot(np.array([i for i in range(850)]), eIMFs[n], 'g')
    plt.ylabel("eIMF %i" %(n+1))
    plt.locator_params(axis='y', nbins=5)

plt.xlabel("Time [s]")
plt.tight_layout()
# plt.savefig('primary_IMFs', dpi=120)
plt.show()

# 任意分量组合
def sum_IMFs(eIMFS, IMFs_index):
    sum_list = []
    for j in range(eIMFs.shape[1]):
        sum_imfs = 0
        for i in IMFs_index:
            sum_imfs += eIMFs[i][j]
        sum_list.append(sum_imfs)
    return np.array(sum_list)
 
IMFs_index = [i for i in range(4)]        #某一重构分量的组成成分
high_IMFs_sum = sum_IMFs(eIMFs, IMFs_index)
plt.plot(np.array([i for i in range(850)]), df.gas_use[0:850].values, 'r')
plt.plot(np.array([i for i in range(850)]), high_IMFs_sum)

# 相关系数计算温度与重构后的分量
temp = df.ave_temp[0:850]
df_tmp = pd.DataFrame({'a':temp,'b':high_IMFs_sum})
print('IMF分量序号：', IMFs_index, '\nPearson系数：', df_tmp.a.corr(df_tmp.b))
