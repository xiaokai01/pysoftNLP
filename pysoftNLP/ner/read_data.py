# -*- coding: utf-8 -*-
# @Time    : 2020/9/21-16:06
# @Author  : 贾志凯
# @File    : read_data.py
# @Software: win10  python3.6 PyCharm
# import pandas as pd
# data = pd.read_table('C:/Users/Administrator/Desktop/a.txt', header=None, encoding='utf-8', sep='\t')
# data.columns = ['标题', '内容']
# data['nr'] = data['标题'] + data['内容']
# data['te'] = ''
# print(data['nr'][0])

#
import re

text = '根据伊泰集团2020年9月13日18点更新的销售价格显示，伊泰集团较上周五调整部分煤矿煤炭坑口价格-30～20元/吨，具体如下：          煤矿名称 产品名称 价格（元/吨） 较9月11日调整（元/吨）'
print(re.compile('集团|公司').findall(text))

# text = '丰田'
# pattern = '丰田|丰田曾宣布与日本宇宙航空研究机构（JAXA）联手开发未来能够在月球上运动的燃料电池六轮月球车'
# m = re.search(pattern, text)
# print('Search     :', m)
# s = re.fullmatch(pattern, text)
# print('Full match :', s)
# if s:
#     print(text)