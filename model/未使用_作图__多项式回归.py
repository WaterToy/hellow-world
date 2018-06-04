import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures


#########################################################################
#                                                                              二项式回归
# 预测函数
def linear_mod(X_parameters, Y_parameters, predict_value):
	lin_reg_2 = linear_model.LinearRegression()
	lin_reg_2.fit(X_poly, Y_parameters)
	predict_outcome = lin_reg_2.predict(predict_value)
	predictions = {}
	predictions['intercept'] = lin_reg_2.intercept_
	predictions['coefficient'] = lin_reg_2.coef_
	predictions['predicted_value'] = predict_outcome
	return predictions

# 读取数据集导入数据
df = pd.read_excel('D:\\20180124_小论文\\20171123数据整理\武汉\居&商_武汉20150101~20170823.xlsx')
datasets_X = df.ave_temp
datasets_Y = df.gas_use
test_forecast = df.ave_temp

length = len(datasets_X)
datasets_X = np.array(datasets_X).reshape([length,1])
datasets_Y = np.array(datasets_Y)


minX = min(datasets_X)
maxX = max(datasets_X)
X = np.arange(minX,maxX).reshape([-1,1])
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(datasets_X)
lin_reg_2 = linear_model.LinearRegression()
lin_reg_2.fit(X_poly, datasets_Y)

predict_outcome = lin_reg_2.predict(X_poly)
predictions = {}
predictions['intercept'] = lin_reg_2.intercept_
predictions['coefficient'] = lin_reg_2.coef_
predictions['predicted_value'] = predict_outcome
# 图像中显示
plt.scatter(datasets_X, datasets_Y, color = 'blue')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'red', linewidth=4)
plt.xlabel('Temperature')
plt.ylabel('Gas_use')
plt.show()