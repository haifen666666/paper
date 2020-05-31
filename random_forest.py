#-*-coding:utf-8-*-
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split,KFold
import datetime

tt = pd.read_csv('final_train.csv')
area = tt['area_id']
#tt['answer'] = tt['answer'] -1
answer=tt[['answer']]

del tt['answer']
del tt['area_id']
#tt.replace(np.inf,np.nan,inplace=True)
#tt.replace(-np.inf,np.nan,inplace=True)
tt.replace(np.inf,100000,inplace=True)
tt.replace(-np.inf,-100000,inplace=True)
tt.fillna(tt.mean(),inplace=True)

x_train, x_test, y_train, y_test=train_test_split(tt, answer, test_size=0.1, random_state=8)
features1 = x_train.values
features2 = x_test.values

answer1 = y_train.answer.values
answer2 = y_test.answer.values


rf = RandomForestClassifier(n_estimators=100,bootstrap = True, oob_score = True, criterion = 'gini',
                            n_jobs=-1,random_state=8)

start = datetime.datetime.now()
rf.fit(features1,answer1)
end = datetime.datetime.now()
print('user %d seconds'%(end - start).seconds)
print('oob_score:\t',rf.oob_score_)
print ("Accuracy:\t", (answer2 == rf.predict(features2)).mean())
print ("Total Correct:\t", (answer2 == rf.predict(features2)).sum())


