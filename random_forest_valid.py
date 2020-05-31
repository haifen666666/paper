#-*-coding:utf-8-*-
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split,KFold

tt = pd.read_csv('final_user_features.csv')
tt['answer'] = tt['answer'] -1
tt.replace(np.inf,100000,inplace=True)
tt.replace(-np.inf,-100000,inplace=True)
tt.fillna(tt.mean(),inplace=True)

area2 = pd.read_csv('valid_area2.csv')
#area2 = area2[area2['area_id']<199990]
area2 = list(area2['area_id'])

x_train=tt.loc[~tt.area_id.isin(area2)]
x_test=tt.loc[tt.area_id.isin(area2)]

print(x_train.shape, x_test.shape)
y_train=x_train[['answer']]
del x_train['answer']
y_test=x_test[['answer']]
#y_test.reset_index(drop=True, inplace=True)
del x_test['answer']

stacking=pd.DataFrame()
stacking['area_id']=x_test['area_id']
stacking.reset_index(inplace=True, drop=True)
stacking['answer'] = y_test
del x_train['area_id']
del x_test['area_id']

features1 = x_train.values
features2 = x_test.values

answer1 = y_train.answer.values
answer2 = y_test.answer.values


rf = RandomForestClassifier(n_estimators=120,bootstrap = True, oob_score = True, criterion = 'gini',
                            n_jobs=-1,random_state=888)
rf.fit(features1,answer1)

valid_ans = pd.DataFrame(rf.predict_proba(features2),
                         columns=['rf1','rf2','rf3','rf4','rf5','rf6','rf7','rf8','rf9'])
stacking=pd.concat([stacking, valid_ans], axis=1)
stacking['pred'] = rf.predict(features2)
stacking.to_csv('user_answer_rf.csv',index=False)

print('tt3')

print('oob_score:\t',rf.oob_score_)

print ("Accuracy:\t", (answer2 == rf.predict(features2)).mean())
print ("Total Correct:\t", (answer2 == rf.predict(features2)).sum())


