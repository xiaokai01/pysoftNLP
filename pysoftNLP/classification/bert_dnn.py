# -*- coding: utf-8 -*-
# @Time    : 2020/11/4-14:32
# @Author  : 贾志凯    15716539228@163.com
# @File    : bert_dnn.py
# @Software: win10  python3.6 PyCharm
import pandas as pd
import numpy as np
# from pysoftNLP.classification.load_data import train_df, test_df
from keras.utils import to_categorical
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, BatchNormalization, Dense,Dropout,SeparableConv1D,Embedding,LSTM
from pysoftNLP.bert.extract_feature import BertVector
import time
import os
#读取文件
def read_data(train_data,test_data):
    file_in = 'D:\pysoftNLP_resources\classification\data'
    train_data = os.path.join(file_in,train_data)
    test_data = os.path.join(file_in,test_data)
    train_df = pd.read_csv(train_data)
    train_df.columns = ['id', 'label', 'text']
    test_df = pd.read_csv(test_data)
    test_df.columns = ['id', 'label', 'text']
    return train_df, test_df

train_data = 'x_tr_863.csv'
test_data = 'x_te_863.csv'
train_df, test_df = read_data(train_data,test_data)
args = {'encode': 'bert', 'sentence_length': 50, 'num_classes': 9, 'batch_size': 128, 'epochs': 100}
def train(train_df,test_df,args):
    out_path = 'D:\pysoftNLP_resources\classification\models'
    print('encoding开始！')
    star_encod_time = time.time()
    bert_model = BertVector(pooling_strategy="REDUCE_MEAN", max_seq_len=args['sentence_length'])  # bert词向量
    f = lambda text: bert_model.encode([text])["encodes"][0]
    train_df['x'] = train_df['text'].apply(f)
    test_df['x'] = test_df['text'].apply(f)
    end_encod_time = time.time()
    print("encoding时间：%s" % (end_encod_time - star_encod_time))
    x_train = np.array([vec for vec in train_df['x']])
    x_test = np.array([vec for vec in test_df['x']])
    y_train = np.array([vec for vec in train_df['label']])
    y_test = np.array([vec for vec in test_df['label']])
    print('x_train: ', x_train.shape)

    y_train = to_categorical(y_train, args['num_classes'])
    y_test = to_categorical(y_test, args['num_classes'])

    x_in = Input(shape=(768,))
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
    x_out = Dense(args.num_classes, activation="softmax")(x_out)
    model = Model(inputs=x_in, outputs=x_out)
    print(model.summary())

    model.compile(loss='categorical_crossentropy',  # categorical_crossentropy
                  optimizer=Adam(),  # adam
                  metrics=['accuracy'])

    # 模型训练以及评估
    model.fit(x_train, y_train, batch_size=args['batch_size'], epochs=args['epochs'])
    wenj = '863_classify_768' + '_' + str(args['sentence_length']) + '_' + str(args['num_classes']) + '_' + str(args['batch_size']) + '_' + str(args['epochs']) + '.h5'
    out_path = os.path.join(out_path, wenj)
    model.save(out_path)

    t3 = time.time()
    print("训练时间：%s" % (t3 - end_encod_time))
    print(model.evaluate(x_test, y_test))
    t4 = time.time()
    print('模型验证时长:', t4 - t3)
train(train_df,test_df,args)