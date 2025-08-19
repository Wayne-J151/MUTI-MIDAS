多元混频MIDAS（Mixed Data Sampling）模型是一种经济计量学模型，用于处理不同频率的数据。MIDAS模型的主要特点是能够结合高频率和低频率的数据，以预测低频率的经济变量。
这种方法在处理实际经济和金融数据时特别有用，因为这些数据通常以不同的频率出现。
个人理解是分为降频和降维两个部分
1、降频：利用了滞后因子多项式，例如almon多项式、beta多项式、指数almon多项式等方法
2、降维：利用了PCA\动态因子\VAR等方法
然后多因子模型的建立就是将降频和降维的结果进行组合并使用多元线性模型进行参数估计
其中关键的点就是同时进行降维、降频、多元线性模型中的参数估计
具体的内容可以看国泰君安的《金融工程专题_MIDAS：混频数据预测通用框架》
以下是我的实现效果
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

#引入写的包
from midas.mix import mix_freq3
from midas.adl import estimate3, forcast3,analyse3
from midas.weights import polynomial_weights
#读取数据
df = pd.read_csv('../data/CPI当月同比等_20230723_220851.csv', index_col=0, parse_dates=True,encoding='gbk')
#对df进行标准化处理，注意有的地方为nan，方便后续做归因分析。
df = df.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))#若需要得到准确的预测值，这一步需要去掉
#检查数据
df.tail(10)
CPI
28种重点监测蔬菜
6种重点监测水果
柴油0
猪肉
date
2023-06-19
NaN
0.218265
0.524030
NaN
-0.192071
2023-06-20
NaN
0.221241
0.530927
NaN
-0.194518
2023-06-21
NaN
0.212313
0.512536
NaN
-0.190439
2023-06-23
NaN
NaN
NaN
0.179334
NaN
2023-06-25
NaN
0.221241
0.475754
NaN
-0.193158
2023-06-26
NaN
0.203384
0.491846
NaN
-0.194518
2023-06-27
NaN
0.212313
0.466559
NaN
-0.195877
2023-06-28
NaN
0.209337
0.482651
NaN
-0.200227
2023-06-29
NaN
0.200408
0.461961
NaN
-0.199412
2023-06-30
-0.314743
0.194456
0.464260
0.191777
-0.198868
#划分训练集与测试集

"""
df: 输入的 DataFrame 数据。第一列是低频数据、其他是高频数据。直接从ifind里面读取数据，然后把末尾的来源去掉。
xlag_list: 一个列表，表示用于预测的不同滞后。第一列一定是1，然后后面的列可以自己定义，比如第一个列为月，第二个列为日，第三个列为周，那么就是[1,21,4]
ylag: 低频数据的滞后期数。用于存在自回归现象，但是不建议开启。
test_start_position_rate: 测试数据开始的位置。如果是小数，比如0.8，那么就是从前面开始的80%作为训练集，后面的20%进行预测。如果是整数，那就是指定预测多少期低频数据。
"""
lf_data_train,lf_data_lag_train,x_train,lf_data_test,lf_data_lag_test,x_test = mix_freq3(df, xlag_list = [1,21,21,4,21],ylag = 0, test_start_position_rate = 0.8)
#参数估计
"""
lf_data: 低频数据。
x_train: 高频数据的训练集。
lf_data_lag: 低频数据的滞后值。
"""
opt_res = estimate3(lf_data_train,x_train,lf_data_lag_train)
#预测
"""
x_test: 高频数据的测试集。
lf_data_lag: 低频数据的滞后值。
opt_res: 优化结果。
"""
pre_yl = forcast3(x_test,lf_data_lag_test,opt_res,lf_data_test)
analyse3(pre_yl,lf_data_test)


添加图片注释，不超过 140 字（可选）

根据趋势线，我们选择相信cpi将会在6月有一个下降的趋势
接下来我们生成滚动预测的趋势预测图
predict_res = pd.Series([])
for i in tqdm(range(1,80)):#建议设置滚动时，数量设置为为数据比较完整部分（本例为80），因为可能有的高频数据不完整，会没有预测结果

    lf_data_train,lf_data_lag_train,x_train,lf_data_test,lf_data_lag_test,x_test = mix_freq3(df, xlag_list = [1,21,21,4,21],ylag = 0, test_start_position_rate = i)
    opt_res = estimate3(lf_data_train,x_train,lf_data_lag_train)
    predict_tmp = forcast3(x_test,lf_data_lag_test,opt_res,lf_data_test)
    predict_res = predict_res.combine_first(predict_tmp)
#画出predict_res与real_res的对比图
analyse3(predict_res,df.iloc[:,0].dropna())

添加图片注释，不超过 140 字（可选）
发现在21年年初，预测与现实的趋势出现了较大的背离，在知道cpi与猪肉相关性很大的情况下，我们选择画出cpi和猪肉的折线图进一步探究
df[['CPI','猪肉']].plot(figsize=(15,4), style=['o','--'])
<Axes: xlabel='date'>

添加图片注释，不超过 140 字（可选）
可以看出确实是在21年猪肉价格和cpi的趋势出现了较大背离， 而且这种背离在历史上也是比较少见的， 所以可以认为这次猪肉价格的上涨是由于猪肉供应量的减少， 而不是由于通货膨胀导致的。

需要具体midas包的可以评论区联系我
