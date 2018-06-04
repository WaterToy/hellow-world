import pandas as pd
from datetime import datetime, timedelta

def holidays_generate(): # 节假日定义

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
        'ds': ['2017-01-28'],
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
    return holidays

def date_list_generate(startdate, enddate):
	days_count = (enddate - startdate).days
	list = []
	for i in range(days_count):
		list.append(startdate + timedelta(days=i))
	
	
def date_property_generate(holidays_df, start=[2015, 1, 1], end=[2017, 8, 24]): # 生成指定形式的日期dataframe,[['date', 'date_property']]
	# 生成一个 DataFrame，包含一列日期
	startdate, enddate = datetime(start[0], start[1], start[2]), datetime(end[0], end[1], end[2])
	days_count = (enddate - startdate).days
	date_list = [(startdate+timedelta(days=i)) for i in range(days_count)]
	date_frame = pd.DataFrame({'date':date_list})
	# 生成一个空的 DataFrame与date_frame合并
	zero_list = [0 for i in range(days_count)]
	zero_frame = pd.DataFrame({'is_holiday':zero_list})
	date_property = pd.concat((date_frame, zero_frame), axis=1)
	# 处理自己定义的节假日，提取成一个list
	holidays_list=[]
	holidays_df = holidays_df.reset_index()
	holidays_df.drop(holidays_df.columns[[0, 2]], axis=1, inplace=True)
	holidays_values = holidays_df.values
	row_num = holidays_values.shape[0]
	for i in range(row_num):
		holidays_list.append([(datetime.strptime(holidays_values[i][0], "%Y-%m-%d")+timedelta(days=j)) for j in range(holidays_values[i][1], 0, 1)])
		holidays_list.append([(datetime.strptime(holidays_values[i][0], "%Y-%m-%d")+timedelta(days=j)) for j in range(0, holidays_values[i][2]+1, 1)])
	new_holidays_list = []
	for i in range(len(holidays_list)):
		new_holidays_list = new_holidays_list + holidays_list[i]
	# 提取holiday信息，生成新的dataframe
	for i in new_holidays_list:
		date_property.ix[date_frame.date==i,'is_holiday']=1
	return date_property