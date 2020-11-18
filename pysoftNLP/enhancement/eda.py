# 文本分类的简易数据扩充技术（数据增强）
# Jason Wei and Kai Zou

import random
from random import shuffle
from pyhanlp import *
from pysoftNLP.enhancement.ciLin import CilinSimilarity # 使用基于信息内容的算法来计算词语相似度               基于哈工大同义词词林扩展版计算语义相似度

random.seed(1)

#stop words list
stop_words = set()

def stopword_init():
    basis = 'D:\pysoftNLP_resources\enhancement'
    stopwords_file = '哈工大停用词表.txt'
    model_path = os.path.join(basis, stopwords_file)
    with open(model_path, 'r',encoding='utf-8') as f:
        for line in f.readlines():
            stop_words.add(line.strip())

    stopwords_file = '百度停用词表.txt'
    model_path = os.path.join(basis, stopwords_file)
    with open(model_path, 'r',encoding='utf-8') as f:
        for line in f.readlines():
            stop_words.add(line.strip())
			
	#stopwords_file = '中文停用词表.txt'
    #with open(stopwords_file, 'r',encoding='utf-8') as f:
    #    for line in f.readlines():
    #        stop_words.add(line.strip())

    print("已经初始化停用词表词个数: ", len(stop_words))

stopword_init()
synonym_handler =  CilinSimilarity()

#pyhanlp进行分词
def get_segment(line):
    HanLP.Config.ShowTermNature = False
    StandardTokenizer = JClass("com.hankcs.hanlp.tokenizer.StandardTokenizer")
    segment_list = StandardTokenizer.segment(line)
    terms_list = []
    for terms in segment_list :
        terms_list.append(str(terms))
	#print(terms_list)
    return terms_list

########################################################################
# Synonym substitution 同义词替换
# Replace n words in the sentence with synonyms from wordnet (用wordnet中的同义词替换句子中的n个单词)
########################################################################


def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words])) #将单词不在停用词的词语形成列表
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms)) #随机选择一个词语
            new_words = [synonym if word == random_word else word for word in new_words]
            print("replaced", random_word, "with", synonym)
            num_replaced += 1
        if num_replaced >= n: #only replace up to n words
            break

    #this is stupid but we need it, trust me
    sentence = ' '.join(new_words)
	#print(sentence,'111111111111111111111111111')
    new_words = sentence.split(' ')
	#print('222222222222222222222222222',new_words)
    return new_words

def get_synonyms(word):
    synonyms = set()
    if word not in synonym_handler.vocab:
        print(word, '未被词林收录！')
    else:
		
        codes = synonym_handler.word_code[word]
        for code in codes:
            key = synonym_handler.code_word[code]
            synonyms.update(key)
        if word in synonyms:
            synonyms.remove(word)

    return list(synonyms)

########################################################################
# Random deletion(随机删除)
# Randomly delete words from the sentence with probability p (用概率p随机删除句子中的单词)
########################################################################

def random_deletion(words, p):

    #obviously, if there's only one word, don't delete it (显然，如果只有一个词，就不要删除它)
    if len(words) == 1:
        return words

    #randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    #if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    return new_words

########################################################################
# Random swap （随机交换）
# Randomly swap two words in the sentence n times （在句子中随机交换两个单词n次）
########################################################################

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words

def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
    return new_words

########################################################################
# Random insertion（随机插入）
# Randomly insert n words into the sentence
########################################################################

def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words

def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)

########################################################################
# main data augmentation function
########################################################################

def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):

    words = get_segment(sentence) # 分词

    num_words = len(words)

    augmented_sentences = []
    num_new_per_technique = int(num_aug/4)+1    #使用几种技术（目前是四种，所以除以4）
    n_sr = max(1, int(alpha_sr*num_words))      #几种技术中替换的单词数
    n_ri = max(1, int(alpha_ri*num_words))
    n_rs = max(1, int(alpha_rs*num_words))

    #sr（随机交换）
    for _ in range(num_new_per_technique):
        a_words = synonym_replacement(words, n_sr)  #
        augmented_sentences.append(''.join(a_words))

    #ri（随机插入）
    for _ in range(num_new_per_technique):
        a_words = random_insertion(words, n_ri)
        augmented_sentences.append(''.join(a_words))

    #rs
    for _ in range(num_new_per_technique):
        a_words = random_swap(words, n_rs)
        augmented_sentences.append(''.join(a_words))

    #rd
    for _ in range(num_new_per_technique):
        a_words = random_deletion(words, p_rd)
        augmented_sentences.append(''.join(a_words))

    augmented_sentences = [get_segment(sentence) for sentence in augmented_sentences]

    shuffle(augmented_sentences)



    #trim so that we have the desired number of augmented sentences （修剪以获得所需数量的增广句子）
    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

    augmented_sentences = [''.join(sentence) for sentence in augmented_sentences]
    #append the original sentence
    augmented_sentences.append(sentence)

    return list(set(augmented_sentences))

