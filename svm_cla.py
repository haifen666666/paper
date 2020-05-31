import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.metrics import *
#########################################################################################
# 模型 OneVsRestClassifier
# 元分类器 svm.LinearSVC
#########################################################################################
def train_svm(filename):
    train = pd.read_csv(filename)
    #train = shuffle(train)
    #train = shuffle(train)
    train.replace(np.inf, 100000, inplace=True)
    train.replace(-np.inf, -100000, inplace=True)
    #train.fillna(0, inplace=True)

    #train = train.replace([np.inf, -np.inf], np.nan)
    #train.fillna(0,inplace=True)
    print(train.shape)

    #tt.replace(np.inf, 100000, inplace=True)
    #tt.replace(-np.inf, -100000, inplace=True)
    #tt.fillna(tt.mean(), inplace=True)

    train['answer'] = train['answer'] - 1
    answer = train[['answer']]
    del train['answer']
    del train['area_id']
    train = (train - train.min()) / (train.max() - train.min())


    x_train, x_test, y_train, y_test = train_test_split(train,answer, test_size = 0.1, random_state = 8)
    x_train.fillna(0,inplace=True)
    x_test.fillna(0,inplace=True)
    #print(train.columns)
    #训练
    #model = OneVsRestClassifier(SVC(kernel='rbf',max_iter=-1,random_state=6), n_jobs=-1)
    model = OneVsRestClassifier(svm.LinearSVC(random_state = 6, verbose = 1,max_iter=200,dual=False),n_jobs=-1)
    #model = OneVsRestClassifier(MultinomialNB(),n_jobs=-1)
    btime = datetime.now()
    model.fit(x_train, y_train)
    print ('all tasks done. total time used:%s s.\n\n'%((datetime.now() - btime).total_seconds()))
    # 保存模型
    joblib.dump(model, 'LinearSVC.pkl')
    joblib.dump(model, 'svm.model')

    # 2、混淆矩阵
    y_pred = model.predict(x_test) # 预测属于哪个类别

    '''
    print(y_pred)
    print(type(y_pred))
    y_pred = pd.DataFrame(y_pred)
    y_test.to_csv('aa.csv',index=False)
    y_pred.to_csv('aaa.csv',index=False)
    y_pred = y_pred.values
    confusion_matrix(y_test.values, y_pred.argmax(axis=1)) # 需要0、1、2、3而不是OH编码格式
    y_pred = y_pred.argsort(axis=1)
    '''
    # 3、经典-精确率、召回率、F1分数
    precision = precision_score(np.array(y_test['answer']),y_pred,average='micro')
    recall = recall_score(np.array(y_test['answer']), y_pred,average='micro')
    f1 = f1_score(np.array(y_test['answer']), y_pred,average='micro')

    print(precision,recall,f1)
    # 保存模型
    joblib.dump(model, 'LinearSVC.pkl')
    joblib.dump(model, 'svm.model')

train_svm('final_train.csv')
