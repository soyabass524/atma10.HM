import pandas as pd

import pandas as pd
import numpy as np

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import pycld2 as cld2

train = pd.read_csv('atmacup10_dataset/train.csv')
test = pd.read_csv('atmacup10_dataset/test.csv')
obj_type = pd.read_csv('atmacup10_dataset/object_collection.csv')

#複数テーブルの結合
cross_object= pd.crosstab(obj_type['object_id'], obj_type['name'])
train = train.merge(cross_object, on='object_id', how='left')
test = test.merge(cross_object,on='object_id', how='left')



cat_cols = ['principal_maker', 'principal_or_first_maker','copyright_holder','acquisition_method','acquisition_credit_line']
for c in cat_cols:
    train.loc[~train[c].isin(test[c].unique()),c] = np.nan
    test.loc[~test[c].isin(train[c].unique()),c] = np.nan






def create_numeric_feature(input_df):
    use_columns = [
        'dating_period',
        'dating_year_early',
        'dating_year_late',
         'Navy Model Room', 'dollhouse', 'drawings',
       'glass', 'jewellery', 'lace', 'musical instruments', 'paintings',
       'paper', 'prints'
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

def title_lan(input_df):
    use_columns = [
        'title'
    ]
    out_df = pd.DataFrame()
    for column in use_columns:
        features = input_df["title"].fillna("").map(lambda x: cld2.detect(x)[2][0][1])
        

        out_df[column] = features
        le = LabelEncoder()
        le.fit(out_df[column])
        out_df[column] = le.transform(out_df[column])
        for index,num in enumerate(out_df[column]):

     
            if num == 11:
                out_df.loc[index]= 1
            else:
                out_df.loc[index] = 0
    
       


    return out_df.add_prefix('TL_')
def hwdt(input):
    words = ['h','w','d','t']
    out_df = pd.DataFrame()
    for word in words:
        out_df[word] = np.zeros(len(input))
    
    for index,values in enumerate(input['sub_title']):

        if type(values) == float:
            continue
        for word in words:
            if word in values:

                out_df.loc[index][word] = 1
            else:

                out_df.loc[index][word] = 0
    return out_df

def hwdtsize(input):
    out_df = pd.DataFrame()
    for axis in ['h', 'w', 't', 'd']:
        column_name = f'size_{axis}'
        size_info = input['sub_title'].str.extract(r'{} (\d*|\d*\.\d*)(cm|mm)'.format(axis)) # 正規表現を使ってサイズを抽出
        size_info = size_info.rename(columns={0: column_name, 1: 'unit'})
        size_info[column_name] = size_info[column_name].replace('', np.nan).astype(float) # dtypeがobjectになってるのでfloatに直す
        size_info[column_name] = size_info.apply(lambda row: row[column_name] * 10 if row['unit'] == 'cm' else row[column_name], axis=1) # 　単位をmmに統一する
        out_df[column_name] = size_info[column_name] 
    return out_df

#ここから新しいデータフレーム

def to_feature(input_df):
    """input_df を特徴量行列に変換した新しいデータフレームを返す.
    """

    processors = [
        create_numeric_feature,
        create_string_length_feature,
        create_count_encoding_feature,
        label_encoder,
        title_lan,
        hwdt,
        hwdtsize
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

train_feat_df.to_csv('new_dataset/train_feat.csv',index=False)
test_feat_df.to_csv('new_dataset/test_feat.csv',index=False)
