import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split,KFold
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import StandardScaler



#cate_list表示类的特征列表
def train(filename,cate_list):
    train = pd.read_csv(filename)
    test = pd.read_csv('test_feature11.csv')

    train.drop(cate_list, axis=1, inplace=True)
    train.replace([np.inf, -np.inf], np.nan, inplace=True)
    train.fillna(0, inplace=True)
    aa = train['area_id']
    bb = train['answer']
    test.drop(cate_list, axis=1, inplace=True)
    test.replace([np.inf, -np.inf], np.nan, inplace=True)
    test.fillna(0, inplace=True)
    cc = test['area_id']
    del train['area_id']
    del train['answer']
    del test['area_id']

    sca = StandardScaler()
    # 先用训练集fit，得到每个特征的均值方差   然后分别用于训练集和验证集
    sca.fit(train)
    x_train_std = sca.transform(train)
    x_test_std = sca.transform(test)

    pca = PCA(n_components=0.8)
    # 同理 先用训练集fit  得到变换的方法 然后将这种变换分别应用于训练集和测试集
    pca.fit(x_train_std)
    x_train_pca = pca.transform(x_train_std)
    x_test_pca = pca.transform(x_test_std)

    x_train_pca = pd.DataFrame(x_train_pca)
    x_train_pca['area_id'] = aa
    x_train_pca['answer'] = bb

    x_test_pca = pd.DataFrame(x_test_pca)
    x_test_pca['area_id'] = cc
    x_train_pca.to_csv('last_train_feature2.csv')
    x_test_pca.to_csv('last_test_feature2.csv')



if __name__ == '__main__':
    categorical_feature1 = ['top1_0','top2_0','top3_0','top1_1','top2_1','top3_1',
                           'top1_2','top2_2','top3_2','top1_3','top2_3','top3_3',
                           'top1_4','top2_4','top3_4','top1_5','top2_5','top3_5',
                           'top1_6','top2_6','top3_6','top1_7','top2_7','top3_7']

    train('train_feature11.csv',categorical_feature1)

