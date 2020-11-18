####  pysoftnlp

#####  --- 河南863软件孵化器有限公司大数据事业部商机雷达项目自然语言处理工具包

#####  --- pysoftNLP是一个提供常用NLP功能的工具包， 宗旨是直接提供方便快捷的解析、词典类的面向中文的工具接口，并提供一步到位的查阅入口。

功能主要包括：

* 中文分词（tokenizer）
* 文本清洗（clean）
* 文本分类（classification）
* 命名实体识别（公司，ner)
* 文本数据增强（enhancement）
* 句义相似度计算（similarities）
* 关键字提取（extraction）

持续更新中...

#### 安装 Installation
* win环境: 
```bash
pip install pysoftNLP==0.0.4 -i https://pypi.python.org/simple
```

* 资源包下载和使用

pysoftNLP_resources 下载之后请放置的D盘中 ,下载地址：链接：https://pan.baidu.com/s/1nAC3c_52ILtZ6WWDCBlpnQ 
提取码：e5qz 
复制这段内容后打开百度网盘手机App，操作更方便哦
```bash
D:\pysoftNLP_resources
```

####  使用 Features

1.分类模型使用
```python
from pysoftNLP.classification import bert_dnn  #分类模型--训练 
train_data = 'x_tr_863.csv'
test_data = 'x_te_863.csv'
train_df, test_df = bert_dnn.read_data(train_data, test_data)
#encode:词向量模型（目前只支持bert） ，sentence_lenth: 50(句子的长度)， num_classes（9分类）
args = {'encode': 'bert', 'sentence_length': 50, 'num_classes': 9, 'batch_size': 128, 'epochs': 100}
bert_dnn.train(train_df, test_df, args)

from pysoftNLP.classification import pre #分类模型--预测
model_name = '863_classify_hy.h5'
label_map = {0:'it',1:'电力热力',2:'化工',3:'环保',4:'建筑',5:'交通 ',6:'教育文化',7:'矿业',8:'绿化',9:'能源',10: '农林' ,11:'市政',12:'水利' ,13:'通信',14:'医疗',15:'制造业'}
texts = ['广西打好“电力牌”组合拳助力工业企业从复产到满产中国新闻网',
         '分别是吕晓雪、唐禄俊、梁秋语、王翠翠、杨兴亮、吕桃桃、张耀夫、郭建波、中国医护服务网',
         '富拉尔基区市场监管局开展《优化营商环境条例》宣传活动齐齐哈尔市人民政府',
         '2020上海（国际）胶粘带与薄膜技术展览会制造交易网'
         ]
pre.predict(model_name,texts,label_map)
```

2、数据增强
```python
from pysoftNLP.enhancement import augment
input = 'D:\pysoftNLP_resources\enhancement\Test\Trian_hy.csv'
output = 'D:\pysoftNLP_resources\enhancement\Test\Trian_out.csv'
num_aug = 20 #一条数据可以扩展到多少条
alpha = 0.05 #一条数据量变化的百分比
augment.gen_eda(input,output,alpha,num_aug)
```
3、命名实体识别（公司）
```python
from pysoftNLP.ner import train
args = {'sentence_length': 100, 'batch_size': 512, 'epochs': 20} # 参数
output_path = 'ner_company' #模型的输出
train.train(args, output_path)

#单句预测
from pysoftNLP.ner import pre
text = '这是一个单句' #
model_name = 'ner'
pre.single_sentence(text,model_name)  

#多句预测
out_path = 'D:\pysoftNLP_resources\entity_recognition'
list = ['中广核新能源湖南分公司', '该公司', '中广核新能源公司']
pre.multi_sentence(list,output_path,model_name)
```
4、相似度计算
```python
#文本相似度计算
from pysoftNLP.similarities import similar
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
similar.similar(sentences,test_vec,args,3)
```

5、关键词抽取
```python
from pysoftNLP.extraction import keyword
text = '6月19日,《2012年度“中国爱心城市”公益活动新闻发布会》在京举行。' + \
       '中华社会救助基金会理事长许嘉璐到会讲话。基金会高级顾问朱发忠,全国老龄' + \
       '办副主任朱勇,民政部社会救助司助理巡视员周萍,中华社会救助基金会副理事长耿志远,' + \
       '重庆市民政局巡视员谭明政。'
print(text)
pos = True
seg_list = keyword.seg_to_list(text, pos)
filter_list = keyword.word_filter(seg_list, pos)

print('TF-IDF模型结果：')
keyword.tfidf_extract(filter_list)
print('TextRank模型结果：')
keyword.textrank_extract(text)
print('LSI模型结果：')
keyword.topic_extract(filter_list, 'LSI', pos)
print('LDA模型结果：')
keyword.topic_extract(filter_list, 'LDA', pos)
```





