# -*- coding: utf-8 -*-
# @Time    : 2020/8/10-17:13
# @Author  : 贾志凯
# @File    : train.py
# @Software: win10  python3.6 PyCharm
from pysoftNLP.kashgari.corpus import ChineseDailyNerCorpus
from pysoftNLP.kashgari.tasks.labeling import BiLSTM_CRF_Model,BiGRU_CRF_Model,BiGRU_Model,BiLSTM_Model,CNN_LSTM_Model
import pysoftNLP.kashgari as kashgari
from pysoftNLP.kashgari.embeddings import BERTEmbedding
import os
def train(args,output_path):
    #加载语料库
    train_x, train_y = ChineseDailyNerCorpus.load_data('train')
    valid_x, valid_y = ChineseDailyNerCorpus.load_data('validate')
    test_x, test_y  = ChineseDailyNerCorpus.load_data('test')
    print(f"训练集大小: {len(train_x)}")
    print(f"验证集大小: {len(valid_x)}")
    print(f"测试集大小: {len(test_x)}")
    print(test_x[:1])

    #训练

    bert_embed = BERTEmbedding('D:\pysoftNLP_resources\pre_training file\chinese_L-12_H-768_A-12',
                               task=kashgari.LABELING,
                               sequence_length=args['sentence_length'])



    model = BiLSTM_CRF_Model(bert_embed)

    model.fit(train_x,
              train_y,
              x_validate=valid_x,
              y_validate=valid_y,
              epochs=args['epochs'],
              batch_size=args['batch_size'])
    basis = 'D:\pysoftNLP_resources\entity_recognition'
    model_path = os.path.join(basis, output_path)
    model.save(model_path)
    #评估
    print(model.evaluate(test_x, test_y))

    #预测
    # loaded_model = kashgari.utils.load_model('saved_ner_model')
if __name__ == '__main__':
    args = {'sentence_length': 100, 'batch_size': 512, 'epochs': 20}
    output_path = 'ner_company'
    train(args,output_path)


