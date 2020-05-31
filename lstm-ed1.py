#-*-coding:utf-8-*-
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler,StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import CuDNNLSTM
from numpy import concatenate
from sklearn.model_selection import train_test_split
from keras import optimizers
from  sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.layers import Dropout

seed_num = 6
import random
random.seed(seed_num)
import numpy
numpy.random.seed(seed_num)
from tensorflow import set_random_seed
set_random_seed(seed_num)

file_lis=os.listdir('./train_visit')
cur_num=0  # 当前处理文件数
file_num=len(file_lis)  # 总文件数

all_train = []
all_answer = []
for filename in file_lis:
    print(cur_num,file_num)
    cur_num += 1
    name=list(filename.strip('.npy').split('_'))
    area_id, func_id=list(map(int, name))
    all_answer.append(func_id-1)  #add answer
    file=os.path.join('./train_visit', filename)
    cur_npy=np.load(file)
    cur_npy = cur_npy.T
    print(cur_npy.shape)
    all_train.append(cur_npy.tolist())
all_train = np.array(all_train)
all_answer = np.array(all_answer)
np.save('all_train.npy',all_train)
np.save('all_answer.npy',all_answer)

all_train = np.load('all_train.npy')
all_answer = np.load('all_answer.npy')


train_x = all_train[:30000]
valid_x = all_train[30000:35000]
test_x = all_train[35000:]

#train_y = all_answer[:30000]
#valid_y = all_answer[30000:35000]
test_y = all_answer[35000:]
train_y = pd.DataFrame(all_answer[:30000].tolist(),columns=['label'])
valid_y = pd.DataFrame(all_answer[30000:35000].tolist(),columns=['label'])
#test_y = pd.DataFrame(all_answer[35000:].tolist(),columns=['label'])

train_y = pd.get_dummies(train_y['label']).values
valid_y = pd.get_dummies(valid_y['label']).values
#test_y = pd.get_dummies(test_y['label']).values

maxvalue = train_x.max()
minvalue = train_x.min()
div = maxvalue - minvalue
train_x = (train_x-minvalue)/div
valid_x = (valid_x-minvalue)/div
test_x = (test_x-minvalue)/div



model=Sequential()
model.add(CuDNNLSTM(100, input_shape=(train_x.shape[1], train_x.shape[2])))
#model.add(Dropout(0.2))
model.add(Dense(9, activation='softmax'))  # 输出层
adam=optimizers.adam(lr=0.005)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
# model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True,dropout=dropout))
model.summary()

# fit network
history=model.fit(train_x, train_y, epochs=30, batch_size=72, validation_data=(valid_x, valid_y), verbose=2,
                  shuffle=False)
yhat=model.predict(test_x)
yhat = yhat.argmax(axis = 1)
test_y = np.array(test_y)
plt.figure(figsize=(20,8),dpi=200)
plt.grid(linestyle="--")
ax = plt.gca()
ax.spines['top'].set_visible(False)  # 去掉上边框
ax.spines['right'].set_visible(False)  # 去掉右边框
ax.set_xlabel('epochs',fontsize=24,fontweight='bold')
ax.set_ylabel('accuracy',fontsize=24,fontweight='bold')
x = list(range(1,31))
y1 = np.array(history.history['accuracy'])
y2 = np.array(history.history['val_accuracy'])
plt.plot(x, y1, marker = 'o',color="blue", label="train", linewidth=1.5)
plt.plot(x, y2,marker = 's',  color="red", label="valid", linewidth=1.5)

plt.legend(loc=2, prop={'family':'Times New Roman', 'size':24})
group_labels = range(0,31,5)
plt.xticks(group_labels,fontsize=24, fontweight='bold')  # 默认字体大小为10
plt.yticks(fontsize=24, fontweight='bold')
plt.savefig('lstm2.png',bbox_inches="tight")
plt.savefig('lstm2.svg', format='svg')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
plt.show()
"""
history字典类型，包含val_loss,val_acc,loss,acc四个key值。 
"""

print(type(np.float(accuracy_score(yhat,test_y))))
print(np.float(accuracy_score(yhat,test_y)))
#print('accuracy %f' % float(accuracy_score(yhat, test_y))+0.15)











