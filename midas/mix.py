import datetime
import re
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

def mix_freq3(df, xlag_list ,ylag = 2, test_start_position_rate = .8):
    scaler = MinMaxScaler()

    # Create an empty DataFrame to store the scaled columns
    df_scaled = pd.DataFrame()

    for column in df.columns:
        # Extract the column
        data = df[column]
        
        # Find where the NaNs are
        nan_locations = np.isnan(data)
        
        # Scale non-NaN values
        data_no_nan = data[~nan_locations]
        scaled_data_no_nan = scaler.fit_transform(data_no_nan.values.reshape(-1, 1))

        # Create a new Series to store the scaled column
        new_data = pd.Series(index=data.index)
        new_data[~nan_locations] = scaled_data_no_nan.flatten()
        new_data[nan_locations] = np.nan
        
        # Add the scaled column to the new DataFrame
        df_scaled[column] = new_data

   
    df = df_scaled

    # 对DataFrame进行排序
    df = df.sort_index()
    df.iloc[-1,0] = 0
    
    # 获取第一列的数据，并删除空值
    lf_data = df.iloc[:,0].dropna()
    
    # 获取ylag后的最小日期
    min_date_y = lf_data.index[ylag]
    min_date_x = df.iloc[:,1].dropna().index[xlag_list[1]]
    
    # 循环遍历DataFrame中的每一列，找到最小的xlag对应的日期
    if df.shape[1]>=2:
        for col_index in range(2,df.shape[1]):
            _ = df.iloc[:,col_index].dropna().index[xlag_list[col_index]]
            tmp_min = df.iloc[:,col_index].dropna().index[xlag_list[col_index]]
        
            if tmp_min>min_date_x:
                min_date_x = tmp_min
    
    # 确定开始日期
    if min_date_y < min_date_x:
        min_date_y = next(d for d in list(lf_data.index) if d > min_date_x)
    start_date = min_date_y

    # 确定训练集和测试集的分割点
     
    if test_start_position_rate >=1:
        split_position =int(len(df.iloc[:,0].dropna().index)) - int(test_start_position_rate)-1
        
    else:
        split_position = int(len(df.iloc[:,0].dropna().index)*test_start_position_rate)
    end_date = lf_data.index[split_position]
    forecast_start_date = lf_data.index[split_position+1]

    forecast_end_date = lf_data.index[-1]
    
    # 如果ylag大于0，生成ylags的数据
    ylags = None
    if ylag > 0:
        ylags = pd.concat([df.iloc[:,0].dropna().shift(1) for l in range(1, ylag + 1)], axis=1)
    
    # 循环遍历DataFrame中的每一列，根据xlag_list生成混合频率的数据
    df_x = []
    for col_index in range(1,df.shape[1]):
        rows = []
        for lfdate in lf_data.loc[start_date:forecast_end_date].index:
            
            hf_data = df.iloc[:,col_index].dropna()

            start_hf = hf_data.index.get_indexer([lfdate],method='pad')[0]  
            
            rows.append(hf_data.iloc[start_hf: start_hf - xlag_list[col_index]: -1].values)
        x = pd.DataFrame(data=rows, index=lf_data.loc[start_date:forecast_end_date].index)
        df_x.append(x)
    
    # 分割训练集和测试集
    x_train = []
    x_test = []
    for df in df_x:
        x_train.append(df.loc[start_date:end_date])
        x_test.append(df.loc[forecast_start_date:])
    
    # 返回训练集和测试集的数据
    return(lf_data.loc[start_date:end_date],
            ylags.loc[start_date:end_date] if ylag > 0 else None,
            x_train,
            lf_data[forecast_start_date:],
            ylags[forecast_start_date:] if ylag > 0 else None,
            x_test)

