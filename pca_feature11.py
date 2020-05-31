import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import StandardScaler

train = pd.read_csv('train_feature11.csv')
categorical_feature = ['top1_0','top2_0','top3_0','top1_1','top2_1','top3_1',
                        'top1_2','top2_2','top3_2','top1_3','top2_3','top3_3',
                        'top1_4','top2_4','top3_4','top1_5','top2_5','top3_5',
                        'top1_6','top2_6','top3_6','top1_7','top2_7','top3_7','area_id']
train['answer'] = train['answer'] - 1
answer = train[['answer']]
del train['answer']

x_train, x_test, y_train, y_test = train_test_split(train,answer, test_size = 0.2, random_state = 100)

x_train2 = x_train[categorical_feature]
x_train.drop(categorical_feature,axis=1,inplace=True)
x_train.replace([np.inf,-np.inf],np.nan,inplace=True)
x_train.fillna(0,inplace=True)

x_test2 = x_test[categorical_feature]
x_test.drop(categorical_feature,axis=1,inplace=True)
x_test.replace([np.inf,-np.inf],np.nan,inplace=True)
x_test.fillna(0,inplace=True)

sca = StandardScaler()
#先用训练集fit，得到每个特征的均值方差   然后分别用于训练集和验证集
sca.fit(x_train)
x_train_std = sca.transform(x_train)
x_test_std = sca.transform(x_test)

pca = PCA(n_components=0.9)
#同理 先用训练集fit  得到变换的方法 然后将这种变换分别应用于训练集和测试集
pca.fit(x_train_std)
x_train_pca = pca.transform(x_train_std)
x_test_pca = pca.transform(x_test_std)


