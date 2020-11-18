# -*- coding: utf-8 -*-
# author: Jclian91
# place: Pudong Shanghai
# time: 2020-02-12 12:57
import pandas as pd

#
# # 读取txt文件
# def read_txt_file(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         content = [_.strip() for _ in f.readlines()]
#
#     labels, texts = [], []
#     for line in content:
#         parts = line.split()
#         label, text = parts[0], ''.join(parts[1:])
#         labels.append(label)
#         texts.append(text)
#
#     return labels, texts

#
# file_path = 'data/train.txt'
# labels, texts = read_txt_file(file_path)
# train_df = pd.DataFrame({'label': labels, 'text': texts})
#
# file_path = 'data/test.txt'
# labels, texts = read_txt_file(file_path)
# test_df = pd.DataFrame({'label': labels, 'text': texts})


train_df = pd.read_csv('data/x_tr_863.csv')
train_df.columns = ['id','label','text']
train_df=train_df.drop(['id'],axis=1)
# train_df=train_df[:500]

test_df = pd.read_csv('data/x_te_863.csv')
test_df.columns = ['id','label','text']
test_df = test_df.drop(['id'],axis=1)
# test_df=test_df[:100]

print(train_df.head())
print(test_df.head())

train_df['text_len'] = train_df['text'].apply(lambda x: len(x))
print(train_df.describe())








