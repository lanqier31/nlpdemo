# -*- coding: gbk -*-
# @File  :
# @Date  : 2018/4/22
# @Software: PyCharm

import jieba
import os
import pickle  # �־û�
from numpy import *
from nlpDemo.FileRead import LoadFolders,LoadFiles,readFile,saveFile
from nlpDemo.BunchSave import readBunch,writeBunch,bunchSave
from nlpDemo.cutText import segText
from nlpDemo.TextParse import get_stop_words
from nlpDemo.TextParse import textParse
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer  # TF-IDF����ת����
from sklearn.feature_extraction.text import TfidfVectorizer  # TF_IDF����������
from sklearn.datasets.base import Bunch
from sklearn.naive_bayes import MultinomialNB  # ����ʽ��Ҷ˹�㷨



def getTestSpace(testSetPath, trainSpacePath, stopWordList, testSpacePath,
                 testSpace_path,testSpace_arr_path,trainbunch_vocabulary_path):
    bunch = readBunch(testSetPath)
    # �������Լ�TF-IDF�����ռ�
    testSpace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[],
                      vocabulary={})
    '''
       ��ȡtestSpace
       '''
    testSpace_out = str(testSpace)
    saveFile(testSpace_path, testSpace_out)
    # ����ѵ�����Ĵʴ�
    trainbunch = readBunch(trainSpacePath)
    # ʹ��TfidfVectorizer��ʼ�������ռ�ģ��  ʹ��ѵ�����ʴ�����
    vectorizer = TfidfVectorizer(stop_words=stopWordList, sublinear_tf=True, max_df=0.5,
                                 vocabulary=trainbunch.vocabulary)
    transformer = TfidfTransformer()
    testSpace.tdm = vectorizer.fit_transform(bunch.contents)
    testSpace.vocabulary = trainbunch.vocabulary
    testSpace_arr = str(testSpace.tdm)
    trainbunch_vocabulary = str(trainbunch.vocabulary)
    saveFile(testSpace_arr_path, testSpace_arr)
    saveFile(trainbunch_vocabulary_path, trainbunch_vocabulary)
    # �־û�
    writeBunch(testSpacePath, testSpace)


def bayesAlgorithm(trainPath, testPath, tfidfspace_out_arr_path,
                   tfidfspace_out_word_path, testspace_out_arr_path,
                   testspace_out_word_apth):
    trainSet = readBunch(trainPath)
    testSet = readBunch(testPath)
    clf = MultinomialNB(alpha=0.001).fit(trainSet.tdm, trainSet.label)
    # alpha:0.001 alpha ԽС����������Խ�࣬����Խ��
    # print(shape(trainSet.tdm))  #������ʾ��������
    # print(shape(testSet.tdm))
    '''����bat�ļ�'''
    tfidfspace_out_arr = str(trainSet.tdm)  # ����
    tfidfspace_out_word = str(trainSet)
    saveFile(tfidfspace_out_arr_path, tfidfspace_out_arr)  # ������ʽ��train_set.txt
    saveFile(tfidfspace_out_word_path, tfidfspace_out_word)  # �ı���ʽ��train_set.txt

    testspace_out_arr = str(testSet)
    testspace_out_word = str(testSet.label)
    saveFile(testspace_out_arr_path, testspace_out_arr)
    saveFile(testspace_out_word_apth, testspace_out_word)

    '''�������'''
    predicted = clf.predict(testSet.tdm)
    total = len(predicted)
    rate = 0
    for flabel, fileName, expct_cate in zip(testSet.label, testSet.filenames, predicted):
        if flabel != expct_cate:
            rate += 1
            print(fileName, ":ʵ�����", flabel, "-->Ԥ�����", expct_cate)
    print("erroe rate:", float(rate) * 100 / float(total), "%")


if __name__ == '__main__':
    # ������Լ�
    stopWordList =get_stop_words()
    test_path = r'../DataSet/Syf/test/'
    test_split_path = r'DataSet/Syf/test_split/'
    test_split_dat_path = r'../DataSet/Syf/test_set.dat'
    tfidfspace_dat_path = r'../DataSet/tfidfspace.dat'
    testspace_dat_path = r'../DataSet/Syf/testspace.dat'
    testSpace_path = r'../DataSet/'
    tfidfspace_out_arr_path = r'../DataSet/Syf/tfidfspace_out_arr.txt'
    tfidfspace_out_word_path = r'../DataSet/Syf/tfidfspace_out_word.txt'
    testspace_out_arr_path = r'../DataSet/Syf/testspace_out_arr.txt'
    testspace_out_word_apth = r'../DataSet/Syf/testspace_out_word.txt'
    segText(test_path,
            test_split_path)  # �Բ��Լ������ļ�������ִʽ��
    bunchSave(test_split_path,
              test_split_dat_path)  #
    getTestSpace(test_split_dat_path,
                 tfidfspace_dat_path,
                 stopWordList,
                 testspace_dat_path,
                 testSpace_path,
                 testSpace_arr_path,
                 trainbunch_vocabulary_path)  # ����ִ��ļ���ͣ�ôʣ�����������������ռ�(txt,dat�ļ�����)
    bayesAlgorithm(tfidfspace_dat_path,
                   testspace_dat_path,
                   tfidfspace_out_arr_path,
                   tfidfspace_out_word_path,
                   testspace_out_arr_path,
                   testspace_out_word_apth)

