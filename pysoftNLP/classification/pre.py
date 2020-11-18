# -*- coding: utf-8 -*-
# @Time    : 2020/6/23-15:03
# @Author  : 贾志凯
# @File    : pre.py
# @Software: win10  python3.6 PyCharm
import pandas as pd
import os
import numpy as np
from pysoftNLP.bert.extract_feature import BertVector
from keras.models import load_model
import time
def predict(model_name,text_list,label_map):
    labels = []
    model_dir = 'D:\pysoftNLP_resources\classification\models'
    model_path = os.path.join(model_dir,model_name)
    model = load_model(model_path)
    bert_model = BertVector(pooling_strategy="REDUCE_MEAN", max_seq_len=80)
    for text in text_list:
    # 将句子转换成向量
        t1=time.time()
        vec = bert_model.encode([text])["encodes"][0]
        t2=time.time()
        print("encode时间：%s"%(t2-t1))
        x_train = np.array([vec])
        # 模型预测
        predicted = model.predict(x_train)
        t3 = time.time()
        print("预测时间：%s"%(t3-t2))
        y = np.argmax(predicted[0])
        labels.append(label_map[y])
    print(labels)
    return labels
    # pass
if __name__ == '__main__':
    model_name = '863_classify_hy.h5'
    label_map = {0:'it',1:'电力热力',2:'化工',3:'环保',4:'建筑',5:'交通 ',6:'教育文化',7:'矿业',8:'绿化',9:'能源',10: '农林' ,11:'市政',12:'水利' ,13:'通信',14:'医疗',15:'制造业'}


# load_model = load_model("D:\pysoftNLP_resources\classification\863_classify_hy.h5")
# # 预测语句
    texts = ['广西打好“电力牌”组合拳助力工业企业从复产到满产中国新闻网',
         '分别是吕晓雪、唐禄俊、梁秋语、王翠翠、杨兴亮、吕桃桃、张耀夫、郭建波、中国医护服务网',
         '富拉尔基区市场监管局开展《优化营商环境条例》宣传活动齐齐哈尔市人民政府',
         '2020上海（国际）胶粘带与薄膜技术展览会制造交易网'
         ]
    predict(model_name,texts,label_map)
# labels = []
# bert_model = BertVector(pooling_strategy="REDUCE_MEAN", max_seq_len=80)
# # 对上述句子进行预测
# for text in texts:
#     # 将句子转换成向量
#     t1=time.time()
#     vec = bert_model.encode([text])["encodes"][0]
#     t2=time.time()
#     print("encode时间：%s"%(t2-t1))
#     x_train = np.array([vec])
#     # 模型预测
#     predicted = load_model.predict(x_train)
#     t3 = time.time()
#     print("预测时间：%s"%(t3-t2))
#     y = np.argmax(predicted[0])
#     # print(predicted)
#     print(y)