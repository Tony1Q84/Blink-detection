import numpy as np
from sklearn import svm
import joblib

train_open_txt = open('train_open.txt', 'r')
train_close_txt = open('train_close.txt', 'r')

train = []
labels = []

'''
读取获取的数据，睁眼数据添加标签1
闭眼数据添加标签0，送入SVM支持向量机中进行训练
'''
print('Reading train_open.txt...')
line_ctr = 0
for txt_str in train_open_txt.readlines():
	temp = []
	datas = txt_str.strip()
	datas = datas.replace('[', '')
	datas = datas.replace(']', '')
	datas = datas.split(',')
	print(datas)
	for data in datas:
		data = float(data)
		temp.append(data)

	train.append(temp)
	labels.append(0)

print('Reading train_close.txt...')
ling_str = 0
temp = []
for txt_str in train_close_txt.readlines():
	temp = []
	datas = txt_str.strip()
	datas = datas.replace('[', '')
	datas = datas.replace(']', '')
	datas = datas.split(',')
	print(datas)
	for data in datas:
		data = float(data)
		temp.append(data)

	train.append(temp)
	labels.append(1)

for i in range(len(labels)):
	print("{0} --> {1}".format(train[i], labels[i]))

train_close_txt.close()
train_open_txt.close()

print(train)
print(labels)
clf = svm.SVC(C = 0.8, kernel = 'linear', gamma = 20, decision_function_shape='ovo')
clf.fit(train, labels)
joblib.dump(clf, "ear_svm.m")


print('predicting [[0.34, 0.34, 0.31]]')
res = clf.predict([[0.34, 0.34, 0.31]])
print(res)

print('predicting [[0.19, 0.18, 0.18]]')
res = clf.predict([[0.19, 0.18, 0.18]])
print(res)