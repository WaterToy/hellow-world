import pandas as pd
import numpy as np
import datetime
from keras.models import Sequential
from keras.layers import Dense
# import seaborn as sns
import matplotlib.pyplot as plt

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# 导入数据
df = pd.read_excel('.\\data\\gas_load.xlsx')
df_date_time = []
for i in range(len(df.date_time)):
    df_date_time.append(df.date_time[i]-datetime.datetime(2015,1,1))
for i in range(len(df_date_time)):
    df_date_time[i] = df_date_time[i].days%365
df_tmp = df[['is_holiday', 'ave_temp', 'gas_use']]
df_tmp.gas_use = df_tmp.gas_use / 5962430
df_tmp.insert(0, 'date_time', df_date_time)
df_matrix = df_tmp.as_matrix()
# 分开输入X，输出Y
X = df_matrix[0:761, 0:3]
Y = df_matrix[0:761, 3]
# create model
model = Sequential()
model.add(Dense(12, input_dim=3, init='uniform', activation='tanh'))
model.add(Dense(10, init='uniform', activation='tanh'))
model.add(Dense(1, init='uniform', activation='tanh'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, nb_epoch=1000, batch_size=20,  verbose=2)
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# calculate predictions
predictions = model.predict(X)
# round predictions

target_x = df_matrix[761:791, 0:3]
prediction_x = model.predict(target_x)

plt.figure()
plt.plot(Y)
plt.plot(predictions)
plt.plot(prediction_x)
plt.plot(df_matrix[761:791, 3])
plt.show()
'''
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w+") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model.from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
'''