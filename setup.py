# -*- coding: utf-8 -*-
# @Time    : 2020/10/28-13:48
# @Author  : 贾志凯    15716539228@163.com
# @File    : setup.py
# @Software: win10  python3.6 PyCharm
import setuptools
with open("README.md", "r",encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="pysoftNLP",
    version="0.0.4",
    author="jiazhikai",
    author_email="15716539228@163.com",
    description="Python wrapper for pysoftNLP: Natural language processing project of 863 Software Incubator Co., Ltd",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'tensorflow', #1.14.0
    ],
)


