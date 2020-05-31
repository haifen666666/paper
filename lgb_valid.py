import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from datetime import datetime

x_train = pd.read_csv('user_feature_train_0.98.csv')
x_test = pd.read_csv('user_feature_test_0.98.csv')
y_train = x_train[['answer']]
del x_train['answer']
y_test = x_test[['answer']]
y_test.reset_index(drop=True,inplace=True)
del x_test['answer']
lgb_train = lgb.Dataset(x_train, y_train, free_raw_data=False)
lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train, free_raw_data=False)

params = {'boosting_type': 'gbdt',
              'objective': 'multiclass',
              'metrics': 'multi_logloss',
              'nthread': 10,   #线程数
              'num_class': 9,  #类别数
              'learning_rate': 0.1,
              'num_leaves': 150,
              'max_depth': 16,
              'max_bin': 200, #将feature存入bin的最大值，越大越准，最大255,默认值255
              'subsample_for_bin': 50000, #用于构建直方图数据的数量，默认值为20000,越大训练效果越好，但速度会越慢
              #'subsample': 0.8, #子采样，为了防止过拟合
              #'subsample_freq': 1,  #重采样频率,如果为正整数，表示每隔多少次迭代进行bagging
              'colsample_bytree': 0.8, #每棵随机采样的列数的占比,一般取0.5-1
              'reg_alpha': 0.2, #L1正则化项，越大越保守
              'reg_lambda': 0, #L2正则化项，越大越保守
              'min_split_gain': 0.0,
              'min_child_weight': 1, #默认值为1,越大越能避免过拟合，建议使用CV调整
              'min_child_samples': 10, #alias：min_data_in_leaf 越大越能避免树过深，避免过拟合，但是可能欠拟合 需要CV调整
              'scale_pos_weight': 1, # 类别不均衡时设定,
              }
num_round = 1000

start = datetime.now()
model_train = lgb.train(params,lgb_train,num_round,valid_sets=lgb_eval,
                        early_stopping_rounds = 20)
print('use seconds:\t',(datetime.now()-start).seconds)
pred =model_train.predict(x_test)

pred = [list(x).index(max(x)) for x in pred]
answer = y_test['answer'].tolist()
print(accuracy_score(answer,pred))


