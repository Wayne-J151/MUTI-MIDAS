import numpy as np

from .weights import polynomial_weights

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
