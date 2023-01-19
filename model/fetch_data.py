import numpy as np
import pandas as pd

def fetch_ts_data(file_name: str, date_col: str, target_col: str):

    data_path = f'../data/{file_name}.csv'

    df = pd.read_csv(
        data_path, 
        parse_dates=[date_col],
        index_col=[date_col],
        infer_datetime_format=True,
        dtype={target_col:'float'}
    )

    df = df.loc[:, [target_col]]
    df[target_col] = df[target_col].apply(pd.to_numeric, downcast='float')

    return df 

