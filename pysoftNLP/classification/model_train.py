# -*- coding: utf-8 -*-
# @Time    : 2020/6/11-17:54
# @Author  : 贾志凯
# @File    : model_train.py
# @Software: win10  python3.6 PyCharm
import os
import tensorflow as tf
# # 如果使用GPU训练
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.2  # 程序最多只能占用指定gpu50%的显存
# sess = tf.Session(config = config)

import pandas as pd
import numpy as np
from pysoftNLP.classification.load_data import train_df, test_df
from keras.utils import to_categorical
from keras.models import Model
from keras.optimizers import Adam,SGD
from keras.layers import Input, BatchNormalization, Dense,Dropout,SeparableConv1D,Embedding,LSTM
from pysoftNLP.bert.extract_feature import BertVector
# from keras.layers.recurrent import LSTM,GRU
from sklearn.naive_bayes import MultinomialNB

#读取文件
def read_data(train_data, test_data):
    train_df = pd.read_csv(train_data)
    train_df.columns = ['id', 'label', 'text']
    test_df = pd.read_csv(test_data)
    test_df.columns = ['id', 'label', 'text']
    return train_df,test_df
# train_data, test_data
# train_df,test_df = read_data(train_data, test_data)

import time
# 读取文件并进行转换
t1 =time.time()

bert_model = BertVector(pooling_strategy="REDUCE_MEAN", max_seq_len=80)
print('begin encoding')
f = lambda text: bert_model.encode([text])["encodes"][0]
train_df['x'] = train_df['text'].apply(f)
test_df['x'] = test_df['text'].apply(f)
print('end encoding')
t2 =time.time()
print("encoding时间：%s"%(t2-t1))
x_train = np.array([vec for vec in train_df['x']])
x_test = np.array([vec for vec in test_df['x']])
y_train = np.array([vec for vec in train_df['label']])
y_test = np.array([vec for vec in test_df['label']])
print('x_train: ', x_train.shape)

# Convert class vectors to binary class matrices.
num_classes = 9
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
print(type(x_train),type(y_train),)
# 创建模型
x_in = Input(shape=(1024, ))
# x_out = Dense(4096, activation="relu")(x_in)
# x_out = BatchNormalization()(x_out)
# x_out = Dropout(0.2)(x_out)
# x_out =Embedding(input_dim=100000,output_dim=1024,input_length=1024)(x_in)
# x_out = SeparableConv1D(0.2)(x_out)
# x_out = LSTM(2048)(x_out)
# x_out = Dense(2048, activation="relu")(x_out)
# x_out = BatchNormalization()(x_out)
# x_out = Dropout(0.2)(x_out)
# x_out = Dense(1024, activation="relu")(x_in)
# x_out = BatchNormalization()(x_out)
# x_out = Dropout(0.2)(x_out)
# x_out = Dense(512, activation="relu")(x_out)
# x_out = BatchNormalization()(x_out)
# x_out = Dropout(0.2)(x_out)
# x_out = Dense(256, activation="relu")(x_out)
# x_out = BatchNormalization()(x_out)
# x_out = Dropout(0.2)(x_out)
# x_out = LSTM(1024)(x_in)
x_out = Dense(1024, activation="relu")(x_in)
x_out = BatchNormalization()(x_out)
x_out = Dropout(0.2)(x_out)
x_out = Dense(512, activation="relu")(x_out)
x_out = BatchNormalization()(x_out)
x_out = Dropout(0.2)(x_out)
x_out = Dense(256, activation="relu")(x_out)
x_out = BatchNormalization()(x_out)
x_out = Dropout(0.2)(x_out)
x_out = Dense(128, activation="relu")(x_out)
x_out = BatchNormalization()(x_out)
x_out = Dropout(0.2)(x_out)
x_out = Dense(64, activation="relu")(x_out)
x_out = BatchNormalization()(x_out)
x_out = Dropout(0.2)(x_out)
x_out = Dense(32, activation="relu")(x_out)
x_out = BatchNormalization()(x_out)
x_out = Dropout(0.2)(x_out)
x_out = Dense(16, activation="relu")(x_out)
x_out = BatchNormalization()(x_out)
x_out = Dense(num_classes, activation="softmax")(x_out)
model = Model(inputs=x_in, outputs=x_out)
print(model.summary())

model.compile(loss='categorical_crossentropy',#categorical_crossentropy
              optimizer=Adam(),   #adam
              metrics=['accuracy'])

# 模型训练以及评估
model.fit(x_train, y_train, batch_size=128, epochs=500)
model.save('863_classify_hy_1024_9.h5')
t3 =time.time()
print("训练时间：%s"%(t3-t2))
print(model.evaluate(x_test, y_test))
t4 = time.time()
print(t4-t3)

# class logger(object):
#     def __init__(self,filename):
#         self.terminal = sys.stdout
#         self.log = open(filename,"a")
#     def write(self,message):
#         self.terminal.write(message)
#         self.log.write(message)
#     def flush(self):
#         pass
# sys.stdout = logger("a.log")
# sys.stderr =logger("A.log")


#
# clf = MultinomialNB()
# clf.fit(x_train,y_train)
# # y_pre = clf.predict(x_test)
# from sklearn.model_selection import cross_val_score
# cvs = cross_val_score(clf,x_test,y_test,scoring="accuracy",cv=10)
# print(cvs)
# print(cvs.mean())