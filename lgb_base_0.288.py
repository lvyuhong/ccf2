# -*- coding: utf-8 -*-
# @Time    : 2019-08-23 17:27
# @Author  : Inf.Turing
# @Site    : 
# @File    : lgb_base.py
# @Software: PyCharm

# 不要浪费太多时间在自己熟悉的地方，要学会适当的绕过一些
# 良好的阶段性收获是坚持的重要动力之一
# 用心做事情，一定会有回报

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
import xgboost as xgb
import catboost as ctb

path = '/Users/inf/PycharmProject/kaggle/ccf_car_sales/data/'

train_sales_data = pd.read_csv(path + '/Train/train_sales_data.csv')
train_search_data = pd.read_csv(path + '/Train/train_search_data.csv')
train_user_reply_data = pd.read_csv(path + '/Train/train_user_reply_data.csv')

test = pd.read_csv(path + '/evaluation_public.csv')

data = pd.concat([train_sales_data, test], ignore_index=True)
data = data.merge(train_search_data, 'left', on=['province', 'adcode', 'model', 'regYear', 'regMonth'])
data = data.merge(train_user_reply_data, 'left', on=['model', 'regYear', 'regMonth'])
data['label'] = data['salesVolume']
data['id'] = data['id'].fillna(0).astype(int)
del data['salesVolume'], data['forecastVolum']

num_feat = ['adcode', 'regMonth', 'regYear', 'popularity', 'carCommentVolum', 'newsReplyVolum']
cate_feat = ['bodyType', 'model', 'province']

for i in cate_feat:
    data[i] = data[i].astype('category')
features = num_feat + cate_feat


def get_predict_w(model, data, label='label', feature=[], cate_feature=[], random_state=2018, n_splits=5,
                  model_type='lgb'):
    if 'sample_weight' not in data.keys():
        data['sample_weight'] = 1
    model.random_state = random_state
    predict_label = 'predict_' + label
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    data[predict_label] = 0
    test_index = (data[label].isnull()) | (data[label] == -1)
    train_data = data[~test_index].reset_index(drop=True)
    test_data = data[test_index]

    for train_idx, val_idx in kfold.split(train_data):
        model.random_state = model.random_state + 1

        train_x = train_data.loc[train_idx][feature]
        train_y = train_data.loc[train_idx][label]

        test_x = train_data.loc[val_idx][feature]
        test_y = train_data.loc[val_idx][label]
        if model_type == 'lgb':
            try:
                model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=100,
                          eval_metric='mae',
                          # callbacks=[lgb.reset_parameter(learning_rate=lambda iter: max(0.005, 0.5 * (0.99 ** iter)))],
                          categorical_feature=cate_feature,
                          sample_weight=train_data.loc[train_idx]['sample_weight'],
                          verbose=100)
            except:
                model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=100,
                          eval_metric='mae',
                          # callbacks=[lgb.reset_parameter(learning_rate=lambda iter: max(0.005, 0.5 * (0.99 ** iter)))],
                          # categorical_feature=cate_feature,
                          sample_weight=train_data.loc[train_idx]['sample_weight'],
                          verbose=100)
        elif model_type == 'ctb':
            model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=100,
                      # eval_metric='mae',
                      # callbacks=[lgb.reset_parameter(learning_rate=lambda iter: max(0.005, 0.5 * (0.99 ** iter)))],
                      cat_features=cate_feature,
                      sample_weight=train_data.loc[train_idx]['sample_weight'],
                      verbose=100)
        train_data.loc[val_idx, predict_label] = model.predict(test_x)
        if len(test_data) != 0:
            test_data[predict_label] = test_data[predict_label] + model.predict(test_data[feature])
    test_data[predict_label] = test_data[predict_label] / n_splits
    print(mse(train_data[label], train_data[predict_label]) * 5, train_data[predict_label].mean(),
          test_data[predict_label].mean())

    return pd.concat([train_data, test_data], sort=True, ignore_index=True), predict_label


lgb_model = lgb.LGBMRegressor(
    num_leaves=32, reg_alpha=0., reg_lambda=0.01, objective='mse', metric='mae',
    max_depth=-1, learning_rate=0.05, min_child_samples=20,
    n_estimators=1000, subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
)
data, predict_label = get_predict_w(lgb_model, data, label='label',
                                    feature=features, cate_feature=cate_feat,
                                    random_state=2019, n_splits=5)

data['lgb'] = data[predict_label]

data['forecastVolum'] = data['lgb'].apply(lambda x: 0 if x < 0 else x)
data[data.label.isnull()][['id', 'forecastVolum']].round().astype(int).to_csv(path + '/sub/sub.csv', index=False)

#
# xgb_model = xgb.XGBRegressor(
#     max_depth=5,
#     learning_rate=0.01,
#     objective='reg:linear',
#     eval_metric='mae',
#     subsample=0.8, colsample_bytree=0.5,
#     n_estimators=3000,
#     reg_alpha=0.,
#     reg_lambda=0.001,
#     min_child_weight=50,
#     n_jobs=8,
#     seed=42,
# )
#
# data, predict_label = get_predict_w(xgb_model, data, label='temp_label',
#                                     feature=features, random_state=2018, n_splits=5)
#
# data['xgb'] = data[predict_label]
#
# ctb_params = {
#     'n_estimators': 10000,
#     'learning_rate': 0.02,
#     'random_seed': 4590,
#     'reg_lambda': 0.08,
#     'subsample': 0.7,
#     'bootstrap_type': 'Bernoulli',
#     'boosting_type': 'Plain',
#     'one_hot_max_size': 10,
#     'rsm': 0.5,
#     'leaf_estimation_iterations': 5,
#     'use_best_model': True,
#     'max_depth': 6,
#     'verbose': -1,
#     'thread_count': 4
# }
#
# ctb_model = ctb.CatBoostRegressor(**ctb_params)
#
# data, predict_label = get_predict_w(ctb_model, data, label='temp_label',
#                                     feature=features,
#                                     random_state=2019, n_splits=5, model_type='ctb')
#
# data['ctb'] = data[predict_label]
#
# data['t_label'] = data['lgb_mse'] * 0.4 + data['xgb_mse'] * 0.1 + data['ctb_mse'] * 0.4 + data['lgb_mae'] * 0.1
