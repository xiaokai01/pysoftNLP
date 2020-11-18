# 文本分类的简易数据扩充技术（数据增强）
# Jason Wei and Kai Zou

from pysoftNLP.enhancement.eda import *

# #要从命令行接收的参数
# import argparse
# ap = argparse.ArgumentParser()
# ap.add_argument("--input", required=True, type=str, help="输入需要增强的文件")
# ap.add_argument("--output", required=False, type=str, help="输出增强之后的文件")
# ap.add_argument("--num_aug", required=False, type=int, help="每个原始句子的增量之后的句子数")
# ap.add_argument("--alpha", required=False, type=float, help="每个句子中要更改的单词百分比")
# args = ap.parse_args()
#
# #the output file
# output = None
# #如果没有接收到转化之后存储位置， 就在当前位置新建
# if args.output:
#     output = args.output
# else:
#     from os.path import dirname, basename, join
#     output = join(dirname(args.input), 'eda_' + basename(args.input))
#
# #每个原始句子的增量之后的句子数
# num_aug = 9 #default
# if args.num_aug:
#     num_aug = args.num_aug
#
# #how much to change each sentence
# alpha = 0.1#default
# if args.alpha:
#     alpha = args.alpha

#使用标准扩充生成更多数据（train_orig：原始的文件, output_file：输出文件, alpha：修改比例, num_aug=9）
def gen_eda(train_orig, output_file, alpha, num_aug=9):
    writer = open(output_file, 'w',encoding='utf-8')
    lines = open(train_orig, 'r',encoding='utf-8').readlines()
	#csv格式按行读取，按，分割， 文本数据在第二列
    for i, line in enumerate(lines):
        parts = line[:-1].split(',')
        label = parts[2] #标注
        sentence = parts[1] #文本
        aug_sentences = eda(sentence, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=num_aug)
        for aug_sentence in aug_sentences:
            writer.write(label + "," + aug_sentence + '\n')

    writer.close()
    print("generated augmented sentences with eda for " + train_orig + " to " + output_file + " with num_aug=" + str(num_aug))


# gen_eda(args.input, output, alpha=alpha, num_aug=num_aug)

