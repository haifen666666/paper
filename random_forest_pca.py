#-*-coding:utf-8-*-
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from datetime import datetime

x_train = pd.read_csv('time_feature_train_0.9.csv')
x_test = pd.read_csv('time_feature_test_0.9.csv')
x_train.fillna(0,inplace=True)
x_test.fillna(0,inplace=True)
y_train = x_train[['answer']]
del x_train['answer']
y_test = x_test[['answer']]
del x_test['answer']
features1 = x_train.values
features2 = x_test.values

answer1 = y_train.answer.values
answer2 = y_test.answer.values


rf = RandomForestClassifier(n_estimators=120,bootstrap = True, oob_score = True, criterion = 'gini',
                            n_jobs=-1,random_state=888)
start = datetime.now()
rf.fit(features1,answer1)
print('oob_score:\t',rf.oob_score_)
print ("Accuracy:\t", (answer2 == rf.predict(features2)).mean())
print ("Total Correct:\t", (answer2 == rf.predict(features2)).sum())


