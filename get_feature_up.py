import pandas as pd
import numpy as np
import os
import pickle
from collections import defaultdict

#遍历每个文件夹，拿到该文件夹下每个用户去过各个区域的字典 eg:{uid1:{0:23,1:15...},...}
#这里采用了交叉统计 比如统计文件夹0内的各个user去过哪些地方  则用1-10文件夹统计得到的用户来往字典
def get_train_user_dict(base_path,target):
    cur_dic = dict()
    for i in range(11):
        if i == target:       #这里如果要给哪个文件夹做字典，就屏蔽这个文件夹，不统计他，这样可以避免过拟合  屏蔽文件夹1
            continue
        print('you are handling %d directory'%i)
        #cur_path = base_path + '/train_part_%d/%d' %(i,i) #获取当前文件夹的路径
        cur_path = base_path + '/%d' %i #获取当前文件夹的路径(这里是服务器的地址，上面是本地的地址)
        cur_path_list = os.listdir(cur_path)  #当前文件夹内的所有文件
        all_nums = len(cur_path_list)
        #print('have_file_numbers:',all_nums)
        #cur_dic = dict()
        cur_number = 0
        for filename in cur_path_list:
            print(i,all_nums,cur_number)
            cur_number += 1
            name = list(filename.strip('.txt').split('_'))
            area_id, func_id = list(map(int, name))
            file_name = cur_path + '/' + filename
            with open(file_name,'r') as f:
                for line in f:
                    uid = line.strip().split('\t')[0]  # 拿到某位用户的id
                    aa=line.strip().split('\t')[1]  # 拿到某位用户的来访记录
                    bb=aa.split(',')  # 使用，获取每一天的来访记录
                    try:
                        cur_dic[uid]
                        try:
                            cur_dic[uid][func_id]
                            for k in bb:
                                cc=k.split('&')
                                hours=cc[1].split('|')  # 获取一天中不同小时的来访记录
                                hours=list(map(int, hours))
                                for hour in hours:
                                    if hour < 12:
                                        cur_dic[uid][func_id][hour]+=1
                        except KeyError:
                            cur_dic[uid][func_id]=[0]*12
                            for k in bb:
                                cc=k.split('&')
                                hours=cc[1].split('|')  # 获取一天中不同小时的来访记录
                                hours=list(map(int, hours))
                                for hour in hours:
                                    if hour < 12:
                                        cur_dic[uid][func_id][hour]+=1
                    except KeyError:
                        cur_dic[uid] = dict()
                        cur_dic[uid][func_id] = [0]*12
                        for k in bb:
                            cc=k.split('&')
                            hours=cc[1].split('|')  # 获取一天中不同小时的来访记录
                            hours=list(map(int, hours))
                            for hour in hours:
                                if hour < 12:
                                    cur_dic[uid][func_id][hour] += 1
    #with open('./data/middle_data/user_for_train%d_new.pkl'%i,'wb') as user_dic:
        #pickle.dump(cur_dic,user_dic)
    print('taotao2')
    cur_user_set = set(cur_dic.keys())  #获取字典内的用户集合，方便做交集
    cur_folder_list = []  #存放当前文件夹内关于用户的特征信息
    cur_path = './train_part/%d'%target
    cur_path_list = os.listdir(cur_path)  # 当前文件夹内的所有文件
    all_nums = len(cur_path_list)
    cur_number = 0
    print('have_file_numbers:', all_nums)
    #遍历每个文件
    for filename in cur_path_list:
        print(target,all_nums,cur_number,'a')
        cur_number += 1
        name = list(filename.strip('.txt').split('_'))
        area_id, func_id = list(map(int, name))
        file_name = cur_path + '/' + filename
        user_set = [] #记录这个文件内的用户id，后面转成set
        with open(file_name,'r') as f:
            for line in f:
                uid = line.strip().split('\t')[0]  # 拿到某位用户的id
                user_set.append(uid)
        user_set = set(user_set)  #当前文件内用户id集合
        rep_user_set = user_set & cur_user_set #找到重复的用户
        #记录这些重复用户中，去过各个其他地方的次数（一个人去了多个地方就统计多个）
        record_dic = {}
        for k in range(1,10):
            record_dic[k] = [0] * 12
        if not rep_user_set:
            cur_folder_list.append([area_id,func_id]+[0]*108)
        else:
            for user in rep_user_set:
                for k in range(1,10):
                    try:
                        single = cur_dic[user][k]
                        record_dic[k] = [m+n for m,n in zip(record_dic[k],single)]
                    except KeyError:
                        pass
            temp_list = []
            for k in range(1,10):
                temp_list += record_dic[k]

            cur_folder_list.append([area_id,func_id]+temp_list)
    feature = pd.DataFrame(cur_folder_list,columns=['area_id','answer']+[k for k in range(108)])
    feature.to_csv('./mid_part/train_feature_%d_up.csv'%target,index=False)
    print('you have finished %d folder'%target)


if __name__ == '__main__':
    base_path = './train_part'
    get_train_user_dict(base_path,3)


