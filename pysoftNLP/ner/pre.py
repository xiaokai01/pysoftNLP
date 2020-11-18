# -*- coding: utf-8 -*-
# @Time    : 2020/8/10-17:42
# @Author  : 贾志凯
# @File    : pre.py
# @Software: win10  python3.6 PyCharm
import pysoftNLP.kashgari as kashgari
import re
import time
import os
import pandas as pd
def load_model(model_name = 'ner'):
    basis = 'D:\pysoftNLP_resources\entity_recognition'
    model_path = os.path.join(basis, model_name)
    load_start = time.time()
    loaded_model = kashgari.utils.load_model(model_path)
    load_end = time.time()
    print("模型加载时间：",load_end-load_start)
    return loaded_model
#
def cut_text(text, lenth):
    textArr = re.findall('.{' + str(lenth) + '}', text)
    textArr.append(text[(len(textArr) * lenth):])
    return textArr

def extract_labels(text, ners):
    ner_reg_list = []
    if ners:
        new_ners = []
        for ner in ners:
            new_ners += ner
        for word, tag in zip([char for char in text], new_ners):
            if tag != 'O':
                ner_reg_list.append((word, tag))

    # 输出模型的NER识别结果
    labels = {}
    if ner_reg_list:
        for i, item in enumerate(ner_reg_list):
            if item[1].startswith('B'):
                label = ""
                end = i + 1
                while end <= len(ner_reg_list) - 1 and ner_reg_list[end][1].startswith('I'):
                    end += 1
                ner_type = item[1].split('-')[1]
                if ner_type not in labels.keys():
                    labels[ner_type] = []
                label += ''.join([item[0] for item in ner_reg_list[i:end]])
                labels[ner_type].append(label)
    return labels

#文本分段
def text_pattern(text):
    list = ['集团|公司','。|，|？|！|、|;|；|：']
    text = '。。' + text
    text = text[::-1]
    temp = []
    def dfs(text, temp):
        if not text:
            return temp
        pattern_text = re.compile(list[0][::-1]).findall(text)
        if pattern_text:
            text = pattern_text[0] + text.split(pattern_text[0], 1)[1]
            comma = re.compile(list[1]).findall(text)[0]
            res_text = text.split(comma, 1)[0]
            temp.append(res_text[::-1])
            text = text.split(comma, 1)[1]
        else:
            # res.append(temp[:]) <class 'list'>: ['中广核新能源湖南分公司']
            return temp
        dfs(text,temp)
    dfs(text,temp)
    return temp

def final_test(path,model_name):
    import pandas as pd
    data = pd.read_table(path, header=None, encoding='utf-8', sep='\t')
    data = data[:200]
    data.columns = ['标题', '内容']
    data['nr'] = data['标题'] + data['内容']
    data['te'] = ''
    for i in range(len(data)):
        first_text = data['nr'][i].replace(" ", "")
        print("原始文本：",first_text)
        last_text = text_pattern(first_text)
        if not last_text:
            continue
        last = []
        for text_input in last_text:
            texts = cut_text(text_input, 100)
            pre_start = time.time()
            ners = load_model(model_name).predict([[char for char in text] for text in texts])
            pre_end = time.time()
            print("切割文章的预测时间：",pre_end - pre_start)
            print("切割的文章内容：",text_input)
            print("切割文本的BIO结果：",ners)
            labels = extract_labels(text_input, ners)
            res = []
            if labels.__contains__('ORG') and labels.__contains__('LOC'):
                entity = labels['ORG'] + labels['LOC']
            elif labels.__contains__('ORG'):
                entity = labels['ORG']
            elif labels.__contains__('LOC'):
                entity = labels['LOC']
            else:
                entity = []
            for j in entity:
                punc = '~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'
                j = re.sub(r"[%s]+" % punc, "", j)
                if re.fullmatch('集团|公司|子公司|本公司|家公司|分公司|上市公司', j):# j == '公司' or j =='集团' or j == '子公司' or j =='本公司' or j == '家公司' or j =='分公司' or j =='上市公司' or j =='母公司': #re.compile('子公司|本公司|家公司|分公司|上市公司').findall(str(j)) or
                    break
                if re.fullmatch('丰田|华为|苹果|微软|爱立信|阿里|三星|中国联通|中国移动|腾讯|联想|台机电|小米|亚马逊|甲骨文|高通|软银|特斯拉|百度|中石化|中石油', j):#j =='华为' or j =='苹果' or j =='微软' or j=='爱立信' or j=='阿里' or j =='三星' or j =='中国联通' or j =='中国移动' or j =='腾讯' or j =='联想':
                    res.append(j)
                elif re.compile('集团|公司|科技|煤炭|医药|工厂|国际|银行|钢铁|机械').findall(str(j[-2:])): #'集团|有限公司|公司|科技|医药|苹果|华为|谷歌|河南863|富士康'
                    res.append(j)
            res = list(set(res))
            print("各个类型的实体结果：", entity)
            print("集团公司：", res)
            if res:
                last.append('|'.join(res))
        last = list(set(last))
        data['te'][i] = '|'.join(last)
        print('最后的公司结果：',"|".join(last))
    pd.DataFrame(data).to_csv('result/a.csv', index=False)
#单句预测
def single_sentence(sentence,model_name):
    first_text = sentence.replace(" ", "")
    print("原始文本：", first_text)
    last_text = text_pattern(first_text)
    if last_text:
        last = []
        for text_input in last_text:
            texts = cut_text(text_input, 100)
            pre_start = time.time()
            ners = load_model(model_name).predict([[char for char in text] for text in texts])
            pre_end = time.time()
            print("切割文章的预测时间：", pre_end - pre_start)
            print("切割的文章内容：", text_input)
            print("切割文本的BIO结果：", ners)
            labels = extract_labels(text_input, ners)
            res = []
            if labels.__contains__('ORG') and labels.__contains__('LOC'):
                entity = labels['ORG'] + labels['LOC']
            elif labels.__contains__('ORG'):
                entity = labels['ORG']
            elif labels.__contains__('LOC'):
                entity = labels['LOC']
            else:
                entity = []
            for j in entity:
                punc = '~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'
                j = re.sub(r"[%s]+" % punc, "", j)
                if re.fullmatch('集团|公司|子公司|本公司|家公司|分公司|上市公司',
                                j):  # j == '公司' or j =='集团' or j == '子公司' or j =='本公司' or j == '家公司' or j =='分公司' or j =='上市公司' or j =='母公司': #re.compile('子公司|本公司|家公司|分公司|上市公司').findall(str(j)) or
                    break
                if re.fullmatch('丰田|华为|苹果|微软|爱立信|阿里|三星|中国联通|中国移动|腾讯|联想|台机电|小米|亚马逊|甲骨文|高通|软银|特斯拉|百度|中石化|中石油',
                                j):  # j =='华为' or j =='苹果' or j =='微软' or j=='爱立信' or j=='阿里' or j =='三星' or j =='中国联通' or j =='中国移动' or j =='腾讯' or j =='联想':
                    res.append(j)
                elif re.compile('集团|公司|科技|煤炭|医药|工厂|国际|银行|钢铁|机械').findall(
                        str(j[-2:])):  # '集团|有限公司|公司|科技|医药|苹果|华为|谷歌|河南863|富士康'
                    res.append(j)
            res = list(set(res))
            print("各个类型的实体结果：", entity)
            print("集团公司：", res)
            if res:
                last.append('|'.join(res))
        last = list(set(last))
        result = "|".join(last)
        print('最后的公司结果：', result)
        return result

#列表式预测
def multi_sentence(sentencelist,out_path,model_name):
    df_data_output = pd.DataFrame()
    df_data_output['text'] = sentencelist
    df_data_output['ner'] = ''
    for i in range(len(sentencelist)):
        first_text = sentencelist[i].replace(" ", "")
        last_text = text_pattern(first_text)
        if not last_text:
            continue
        last = []
        for text_input in last_text:
            texts = cut_text(text_input, 100)
            ners = load_model(model_name).predict([[char for char in text] for text in texts])
            labels = extract_labels(text_input, ners)
            res = []
            if labels.__contains__('ORG') and labels.__contains__('LOC'):
                entity = labels['ORG'] + labels['LOC']
            elif labels.__contains__('ORG'):
                entity = labels['ORG']
            elif labels.__contains__('LOC'):
                entity = labels['LOC']
            else:
                entity = []
            for j in entity:
                punc = '~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'
                j = re.sub(r"[%s]+" % punc, "", j)
                if re.fullmatch('集团|公司|子公司|本公司|家公司|分公司|上市公司',
                                j):  # j == '公司' or j =='集团' or j == '子公司' or j =='本公司' or j == '家公司' or j =='分公司' or j =='上市公司' or j =='母公司': #re.compile('子公司|本公司|家公司|分公司|上市公司').findall(str(j)) or
                    break
                if re.fullmatch('丰田|华为|苹果|微软|爱立信|阿里|三星|中国联通|中国移动|腾讯|联想|台机电|小米|亚马逊|甲骨文|高通|软银|特斯拉|百度|中石化|中石油',
                                j):  # j =='华为' or j =='苹果' or j =='微软' or j=='爱立信' or j=='阿里' or j =='三星' or j =='中国联通' or j =='中国移动' or j =='腾讯' or j =='联想':
                    res.append(j)
                elif re.compile('集团|公司|科技|煤炭|医药|工厂|国际|银行|钢铁|机械').findall(
                        str(j[-2:])):  # '集团|有限公司|公司|科技|医药|苹果|华为|谷歌|河南863|富士康'
                    res.append(j)
            res = list(set(res))
            # print("各个类型的实体结果：", entity)
            # print("集团公司：", res)
            if res:
                last.append('|'.join(res))
        last = list(set(last))
        df_data_output['ner'][i] = '|'.join(last)
        out_path = os.path.join(out_path, r'result.csv')
        # print('最后的公司结果：', "|".join(last))
        pd.DataFrame(df_data_output).to_csv(out_path, index=False)

path = 'C:/Users/Administrator/Desktop/a.txt'
def prdict(path):
    final_test(path)
    # text_input = input('句子: ').stride()x.drop('',axis = 1)
    '''import re
    # text_input = 'BAT：B指百度，A指阿里巴巴，T指腾讯，是中国互联网du公司百度zhi公司（Baidu），阿里巴巴集团（Alibaba），腾讯公司（Tencent）三大互联网公司首字母的缩写。BAT已经成为中国最大的三家互联网公司'
    # text_input ='“新冠疫情或让印度的大国梦碎。”印度《经济时报》5日以此为题报道称，疫情正在对印度经济造成严重影响。据印度卫生部门统计，截至当地时间6日早，印度过去一天新增新冠肺炎确诊病例90633例，再创历史新高，累计确诊已超411万例，累计死亡70626例。印度即将超过巴西成为仅次于美国的全球确诊病例第二高的国家。'
    # text_input = '工程机械持续火热：国内挖掘机销量连5个月同比增速超50%	当前，工程机械行业持续火热。业内人士认为，国内复工复产和基建、房地产共同带来的需求，是拉动工程机械销量在二季度实现大反弹的主要因素。预计四季度工程机械行业仍将维持高景气，增长势头有望延续。 据最新数据，2020年8月纳入统计的25家挖掘机制造企业共销售各类挖掘机20939台，同比增长51.3%。国内挖掘机销量连续5个月同比增速保持50%以上。今年前8个月销售总量已占到2019年全年销量的89.3%。'
    input = ['中广核新能源湖南分公司','该公司','中广核新能源公司']
    last = []
    for text_input in input:
        texts = cut_text(text_input, 100)
        pre_start= time.time()
        ners = loaded_model.predict([[char for char in text] for text in texts])
        pre_end = time.time()
        print("预测时间：",pre_end - pre_start)
        print("文章内容：",text_input)
        print("BIO结果：",ners)
        labels = extract_labels(text_input, ners)
        res = []
        if labels.__contains__('ORG') and labels.__contains__('LOC'):
            entity = labels['ORG'] + labels['LOC']
        elif labels.__contains__('ORG'):
            entity = labels['ORG']
        elif labels.__contains__('LOC'):
            entity = labels['LOC']
        else:
            entity = []
        for j in entity:
            punc = '~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'
            j = re.sub(r"[%s]+" % punc, "", j)
            if re.fullmatch('集团|公司|子公司|本公司|家公司|分公司|上市公司', j):# j == '公司' or j =='集团' or j == '子公司' or j =='本公司' or j == '家公司' or j =='分公司' or j =='上市公司' or j =='母公司': #re.compile('子公司|本公司|家公司|分公司|上市公司').findall(str(j)) or
                break
            if re.fullmatch('丰田|华为|苹果|微软|爱立信|阿里|三星|中国联通|中国移动|腾讯|联想|台机电|小米|亚马逊|甲骨文|高通|软银|特斯拉|百度|中石化|中石油', j):#j =='华为' or j =='苹果' or j =='微软' or j=='爱立信' or j=='阿里' or j =='三星' or j =='中国联通' or j =='中国移动' or j =='腾讯' or j =='联想':
                res.append(j)
                    # break
            elif re.compile('集团|公司|科技|煤炭|医药|工厂|国际|银行|钢铁|机械').findall(str(j[-2:])): #'集团|有限公司|公司|科技|医药|苹果|华为|谷歌|河南863|富士康'
                res.append(j)
        res = list(set(res))
            # data['te'][i] = '|'.join(res)
        # print("各个类型的实体结果：", labels['ORG'])
        # print(labels,type(labels))
        print("各个类型的实体结果：", entity)
        print("集团公司：", res)
        if res:
            last.append(''.join(res))
    print(last)
    print('最后的公司结果：',"|".join(last))'''


