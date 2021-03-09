import pandas as pd

import pandas as pd
import numpy as np

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

train = pd.read_csv('atmacup10_dataset/train.csv')
test = pd.read_csv('atmacup10_dataset/test.csv')


cat_cols = ['principal_maker', 'principal_or_first_maker','copyright_holder','acquisition_method','acquisition_credit_line']
for c in cat_cols:
    train.loc[~train[c].isin(test[c].unique()),c] = np.nan
    test.loc[~test[c].isin(train[c].unique()),c] = np.nan





#titleは後で言語割り出して使いたい
def create_numeric_feature(input_df):
    use_columns = [
        'dating_period',
        'dating_year_early',
        'dating_year_late'
    ]
    return input_df[use_columns].copy()

def create_string_length_feature(input_df): #タイトルなどの文字数
    out_df = pd.DataFrame()

    str_columns = [
        'title', 
        'long_title',
        'sub_title',
        'more_title',
        'description',
        
        # and more
    ]
    
    for c in str_columns:
        out_df[c] = input_df[c].str.len()

    return out_df.add_prefix('StringLength__')

def create_count_encoding_feature(input_df):
    use_columns = [
        'acquisition_method',
        'principal_maker',
        'principal_or_first_maker',
        'acquisition_credit_line'
        # and more
    ]

    out_df = pd.DataFrame()
    for column in use_columns:
        vc = train[column].value_counts()
        out_df[column] = input_df[column].map(vc)

    return out_df.add_prefix('CE_')

def label_encoder(input_df):
    use_columns = [
        'copyright_holder',
        'acquisition_method',
        'principal_maker',
        'principal_or_first_maker',
        'acquisition_credit_line'   
    ]
    
    out_df = pd.DataFrame()
    for column in use_columns:
        le = LabelEncoder()
        le.fit(input_df[column])
        out_df[column] = le.transform(input_df[column])
    
    return out_df.add_prefix('LE_')

#ここから新しいデータフレーム
def to_feature(input_df):
    """input_df を特徴量行列に変換した新しいデータフレームを返す.
    """

    processors = [
        create_numeric_feature,
        create_string_length_feature,
        create_count_encoding_feature,
        label_encoder
    ]

    out_df = pd.DataFrame()

    for func in tqdm(processors, total=len(processors)):
        _df = func(input_df)

        # 長さが等しいことをチェック (ずれている場合, func の実装がおかしい)
        assert len(_df) == len(input_df), func.__name__
        out_df = pd.concat([out_df, _df], axis=1)

    return out_df

train_feat_df = to_feature(train)
test_feat_df = to_feature(test)

train_feat_df.to_csv('train_feat.csv',index=False)
test_feat_df.to_csv('test_feat.csv',index=False)
#ここからCV
#y_train = np.log1p(train_feat_df[''])