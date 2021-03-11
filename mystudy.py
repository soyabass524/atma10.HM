import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold,StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import mean_squared_error



train = pd.read_csv('atmacup10_dataset/train.csv')
y = train['likes'].values
y = np.log1p(y)

train_feat_df = pd.read_csv('new_dataset/train_feat.csv')
test_feat_df = pd.read_csv('new_dataset/test_feat.csv')
sub = pd.read_csv('atmacup10_dataset/submission.csv')
delete_column = ['LE_copyright_holder']
train_feat_df.drop(delete_column, axis=1, inplace=True)
test_feat_df.drop(delete_column, axis=1, inplace=True)


y_train = y
X_train = train_feat_df
X_test = test_feat_df

y_preds = []
models = []
oof_train = np.zeros((len(X_train),))
cv = KFold(n_splits=5, shuffle=True, random_state=0)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
params = {'num_leaves': 32,
               'min_data_in_leaf': 64,
               'objective': 'regression',
               'max_depth': -1,
               'learning_rate': 0.05,
               "boosting": "gbdt",
               "bagging_freq": 1,
               "bagging_fraction": 0.8,
               "bagging_seed": 0,
               "verbosity": -1,
              'reg_alpha': 0.1,
              'reg_lambda': 0.3,
              'colsample_bytree': 0.7,
              'metric':"rmse",
              'num_threads':6,
         }
categorical_features = ['CE_acquisition_method',
       'CE_principal_maker', 'CE_principal_or_first_maker',
       'CE_acquisition_credit_line', 
       'LE_acquisition_method', 'LE_principal_maker',
       'LE_principal_or_first_maker', 'LE_acquisition_credit_line','TL_title','h','w','d','t', 'Navy Model Room', 'dollhouse', 'drawings',
       'glass', 'jewellery', 'lace', 'musical instruments', 'paintings',
       'paper', 'prints']
y_preds = 0

for fold_id, (train_index, valid_index) in enumerate(kf.split(X_train,y_train)):
    X_tr = X_train.loc[train_index, :]
    X_val = X_train.loc[valid_index, :]
    y_tr = y_train[train_index]
    y_val = y_train[valid_index]
    lgb_train = lgb.Dataset(X_tr, y_tr, categorical_feature=categorical_features)
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train, categorical_feature=categorical_features)
    
    model = lgb.train(params, lgb_train,
                      valid_sets=[lgb_train, lgb_eval],
                      verbose_eval = 200,
                      num_boost_round = 10000,
                      early_stopping_rounds=200)
    
    oof_train[valid_index] = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_preds += y_pred/5
    models.append(model)
pd.DataFrame(oof_train).to_csv('oof_train_kfold.csv', index=False)

y_preds = np.expm1(y_preds)
y_preds[y_preds<0] = 0
sub['likes'] =y_preds
sub.to_csv('first_try.csv', index=False) 
