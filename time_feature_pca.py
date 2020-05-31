import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

cate_list=['top1_0', 'top2_0', 'top3_0', 'top1_1', 'top2_1', 'top3_1',
                      'top1_2', 'top2_2', 'top3_2', 'top1_3', 'top2_3', 'top3_3',
                      'top1_4', 'top2_4', 'top3_4', 'top1_5', 'top2_5', 'top3_5',
                      'top1_6', 'top2_6', 'top3_6', 'top1_7', 'top2_7', 'top3_7','area_id']


#cate_list表示类的特征列表
train = pd.read_csv('train_feature11.csv')
print(train.shape)
train['answer'] = train['answer'] - 1

area2 = pd.read_csv('valid_area2.csv')
area2 = list(area2['area_id'])
x_train = train.loc[~train.area_id.isin(area2)]
x_test = train.loc[train.area_id.isin(area2)]
y_train = x_train[['answer']]
y_train.reset_index(drop=True,inplace=True)
del x_train['answer']
y_test = x_test[['answer']]
y_test.reset_index(drop=True,inplace=True)
del x_test['answer']


#x_train.drop(cate_list, axis=1, inplace=True)
x_train.replace([np.inf, -np.inf], np.nan, inplace=True)
x_train.fillna(0, inplace=True)

#x_test.drop(cate_list, axis=1, inplace=True)
x_test.replace([np.inf, -np.inf], np.nan, inplace=True)
x_test.fillna(0, inplace=True)

sca = StandardScaler()
# 先用训练集fit，得到每个特征的均值方差   然后分别用于训练集和验证集
sca.fit(x_train)
x_train_std = sca.transform(x_train)
x_test_std = sca.transform(x_test)

pca = PCA(n_components=0.95)
# 同理 先用训练集fit  得到变换的方法 然后将这种变换分别应用于训练集和测试集
pca.fit(x_train_std)
x_train_pca = pca.transform(x_train_std)
x_test_pca = pca.transform(x_test_std)

x_train_pca = pd.DataFrame(x_train_pca)
x_train_pca['answer'] = y_train['answer']
x_test_pca = pd.DataFrame(x_test_pca)
x_test_pca['answer'] = y_test['answer']
print(x_train_pca.shape)
print(x_test_pca.shape)
x_train_pca.to_csv('time_feature_train_0.9.csv',index=False)
x_test_pca.to_csv('time_feature_test_0.9.csv',index=False)





