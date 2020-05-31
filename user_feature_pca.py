import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

threshold = 0.98
train = pd.read_csv('final_user_features.csv')
train['answer'] = train['answer'] - 1

train.replace(np.inf,100000,inplace=True)
train.replace(-np.inf,-100000,inplace=True)
train.fillna(train.mean(),inplace=True)

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

sca = StandardScaler()
# 先用训练集fit，得到每个特征的均值方差   然后分别用于训练集和验证集
sca.fit(x_train)
x_train_std = sca.transform(x_train)
x_test_std = sca.transform(x_test)

pca = PCA(n_components=threshold)  ######################################
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
x_train_pca.to_csv('user_feature_train_0.98.csv',index=False)
x_test_pca.to_csv('user_feature_test_0.98.csv',index=False)

