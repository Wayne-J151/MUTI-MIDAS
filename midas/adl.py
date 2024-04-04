import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import *

from scipy.optimize import least_squares

from .weights import polynomial_weights

from .fit import ssr3

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方

def estimate3(lf_data_train,x_train,lf_data_lag_train,model_tpye = "liner",lasso_lambda = 0.1):
    yx_df = [np.ones(len(lf_data_train))]
    for x in x_train:
        weight_method = polynomial_weights("expalmon")
        xw, w = weight_method.x_weighted(x, weight_method.init_params())
        yx_df.append(xw)
    yx_df = pd.DataFrame(data=yx_df).T
    #将xw_df的index设置为lf_data_train的index
    yx_df.index = lf_data_train.index
    #为wx_df增加lf_data_lag_train，注意lf_data_lag_train是个dataframe
    yx_df = pd.concat([yx_df,lf_data_lag_train],axis=1)

    c = np.linalg.lstsq(yx_df, lf_data_train,rcond=None)[0]
    a_ = [0,-1]*len(x_train)
    a_.extend(c)
    if model_tpye == "liner":
        f = lambda v: ssr3(v, lf_data_train, x_train,lf_data_lag_train)
    if model_tpye == "lasso":
        f = lambda v: ssr3_lasso(v, lf_data_train, x_train,lf_data_lag_train,lasso_lambda)
    opt_res = least_squares(f,
                            a_,

                            xtol=1e-5,
                            ftol=1e-5,
                            max_nfev=3000,
                            verbose=0
    )
 
    return opt_res

def ssr3(a, lf_data_train, x_train,lf_data_lag_train):

    yx_df_ = [np.ones(len(lf_data_train))]
    for x_index in range(len(x_train)):
        weight_method = polynomial_weights("expalmon")
        xw, w = weight_method.x_weighted(x_train[x_index], (a[2*x_index],a[2*x_index+1]))
        yx_df_.append(xw)
    yx_df_ = pd.DataFrame(data=yx_df_).T
    #将xw_df的index设置为lf_data_train的index
    yx_df_.index = lf_data_train.index
    #为wx_df增加lf_data_lag_train，注意lf_data_lag_train是个dataframe
    yx_df_2 = pd.concat([yx_df_,lf_data_lag_train],axis=1)
    vector = np.array(a[2*len(x_train):]).reshape(-1, 1)
    
    error = list(lf_data_train - np.dot(yx_df_2, vector).reshape(-1))
    
    return error


def ssr3_lasso(a, lf_data_train, x_train,lf_data_lag_train,lasso_lambda):

    yx_df_ = [np.ones(len(lf_data_train))]
    for x_index in range(len(x_train)):
        weight_method = polynomial_weights("expalmon")
        xw, w = weight_method.x_weighted(x_train[x_index], (a[2*x_index],a[2*x_index+1]))
        yx_df_.append(xw)
    yx_df_ = pd.DataFrame(data=yx_df_).T
    #将xw_df的index设置为lf_data_train的index
    yx_df_.index = lf_data_train.index
    #为wx_df增加lf_data_lag_train，注意lf_data_lag_train是个dataframe
    yx_df_2 = pd.concat([yx_df_,lf_data_lag_train],axis=1)
    vector = np.array(a[2*len(x_train):]).reshape(-1, 1)
    
    error = list(lf_data_train - np.dot(yx_df_2, vector).reshape(-1))
    #add lasso
    error += [sqrt(abs(i*lasso_lambda)) for i in a[2*len(x_train):]]

    return error

def forcast3(x_test,lf_data_lag_test,opt_res,lf_data_test):
    xw_matrix = pd.DataFrame(np.ones(len(x_test[0])))
    for x_index in range(len(x_test)):
        weight_method = polynomial_weights('expalmon')
        xw, _ = weight_method.x_weighted(x_test[x_index].values, [opt_res.x[2*x_index],opt_res.x[2*x_index+1]])
        xw_matrix = pd.concat([xw_matrix,pd.DataFrame(xw)],axis=1)
    xw_matrix.index = x_test[0].index
    vector = opt_res.x.reshape(-1, 1)[2*len(x_test):]
    xy_test = pd.concat([xw_matrix,lf_data_lag_test],axis=1)
    pre_yl = np.dot(xy_test, vector).reshape(-1)
    #给pre_yl设置index为lf_data_test.index
    pre_yl = pd.Series(pre_yl)

    pre_yl.index = lf_data_test.index
    return pre_yl



def analyse3(predict_res,df_series):
    # plt.rcParams['axes.grid'] = True
    # plt.plot(pre_yl)
    # plt.plot(lf_data_test[:-1])
    # #设置标题为
    # plt.title('MIDAS Model')
    # #设置tag
    # plt.legend(['Predicted','Actual'])
    # # plt.xticks( rotation='45')
    # # plt.show()
    # #横坐标斜一点


    # 使用align方法对齐两个序列
    df_series, predict_res = df_series.align(predict_res, join='inner')

    # 计算趋势
    df_diff = df_series.diff().dropna()
    predict_diff = predict_res.diff().dropna()

    # 比较趋势并创建胜利/失败序列
    win_lose = (df_diff * predict_diff >= 0).astype(int)

    # 绘制图形
    plt.figure(figsize=(10, 5))

    dates = win_lose.index
    for i, value in enumerate(win_lose):
        try:
            start_date = dates[i-1]
        except IndexError:  # 如果是最后一个元素
            start_date = dates[i]
        end_date = dates[i]


        if value == 1:
            plt.plot([start_date, end_date], [0, 0], color='red', linewidth=5)
        else:
            plt.plot([start_date, end_date], [0, 0], color='green', linewidth=5)

    plt.rcParams['axes.grid'] = True
    plt.plot(predict_res[1:],label='predict')
    plt.plot(df_series[1:],label='true value')

    #设置tag

    plt.legend()
    #设置标题为
    plt.title('MIDAS Model')


    #计算win_lose大于0的个数
    right_rate = win_lose[win_lose>0].count()/len(win_lose)
    #输出胜率，并限制为小数点后两位
    print('胜率为：%.2f%%' % (right_rate*100))

    plt.show()