# -*- coding: utf-8 -*-
# @Time    : 2020/11/3-13:23
# @Author  : 贾志凯    15716539228@163.com
# @File    : similar.py
# @Software: win10  python3.6 PyCharm
import numpy as np
# from bert_serving.client import BertClient
# bc = BertClient(ip='localhost',check_version=False,port=5555, port_out=5556, check_length=False,timeout=10000)
# topk = 3
#
# sentences = ['逍遥派掌门人无崖子为寻找一个色艺双全、聪明伶俐的徒弟，设下“珍珑”棋局，为少林寺虚字辈弟子虚竹误撞解开。',
#              '慕容复为应召拒绝王语嫣的爱情；众人救起伤心自杀的王语嫣，后段誉终于获得她的芳心。',
#              '鸠摩智贪练少林武功，走火入魔，幸被段誉吸去全身功力，保住性命，大彻大悟，成为一代高僧。',
#              '张无忌历尽艰辛，备受误解，化解恩仇，最终也查明了丐帮史火龙之死乃是成昆、陈友谅师徒所为',
#              '武氏与柯镇恶带着垂死的陆氏夫妇和几个小孩相聚，不料李莫愁尾随追来，打伤武三通',
#              '人工智能亦称智械、机器智能，指由人制造出来的机器所表现出来的智能。',
#              '人工智能的研究是高度技术性和专业的，各分支领域都是深入且各不相通的，因而涉及范围极广。',
#              '自然语言认知和理解是让计算机把输入的语言变成有意思的符号和关系，然后根据目的再处理。']
#
# sentences_vec = bc.encode(sentences)
# print(type(sentences_vec))
# test_vec = bc.encode(['自然语言处理与人工智能'])
# score = np.sum(test_vec * sentences_vec, axis=1) / np.linalg.norm(sentences_vec, axis=1)
# topk_idx = np.argsort(score)[::-1][:topk]
# for idx in topk_idx:
#     print('> 相似度:%s\t相似句子:%s' % (score[idx], sentences[idx]))

from pysoftNLP.bert.extract_feature import BertVector
import pandas as pd

def similar(sentences,test_vec,args,topk):
    bert_model = BertVector(pooling_strategy="REDUCE_MEAN", max_seq_len=args['sentence_length'])  # bert词向量
    f = lambda text: bert_model.encode([text])["encodes"][0]
    sentences_vec = pd.Series(sentences).apply(f)
    test_vec = pd.Series(test_vec).apply(f)
    sentences_vec = np.array([vec for vec in sentences_vec])
    test_vec = np.array([vec for vec in test_vec])
    score = np.sum(test_vec * sentences_vec, axis=1) / np.linalg.norm(sentences_vec, axis=1)
    topk_idx = np.argsort(score)[::-1][:topk]
    for idx in topk_idx:
        print('> 相似度:%s\t相似句子:%s' % (score[idx], sentences[idx]))

if __name__ == '__main__':
    test_vec =  '自然语言处理与人工智能'
    sentences = ['逍遥派掌门人无崖子为寻找一个色艺双全、聪明伶俐的徒弟，设下“珍珑”棋局，为少林寺虚字辈弟子虚竹误撞解开。',
                 '慕容复为应召拒绝王语嫣的爱情；众人救起伤心自杀的王语嫣，后段誉终于获得她的芳心。',
                 '鸠摩智贪练少林武功，走火入魔，幸被段誉吸去全身功力，保住性命，大彻大悟，成为一代高僧。',
                 '张无忌历尽艰辛，备受误解，化解恩仇，最终也查明了丐帮史火龙之死乃是成昆、陈友谅师徒所为',
                 '武氏与柯镇恶带着垂死的陆氏夫妇和几个小孩相聚，不料李莫愁尾随追来，打伤武三通',
                 '人工智能亦称智械、机器智能，指由人制造出来的机器所表现出来的智能。',
                 '人工智能的研究是高度技术性和专业的，各分支领域都是深入且各不相通的，因而涉及范围极广。',
                 '自然语言认知和理解是让计算机把输入的语言变成有意思的符号和关系，然后根据目的再处理。']
    args = {'encode': 'bert', 'sentence_length': 50, 'num_classes': 9, 'batch_size': 128, 'epochs': 100}
    similar(sentences,test_vec,args,3)
