#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 02:20:35 2017

@author: rachit
"""
import random
import nltk
from nltk import word_tokenize
import hashlib
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
def npcollocation(doc):
    blob=TextBlob(doc)
    tokens=word_tokenize(doc)
    np=blob.noun_phrases
    np_tokenize=[]
    for phrase in np:
        np_tokenize.append((phrase,word_tokenize(phrase)))
    for phrase,nptokens in np_tokenize:
        sz=len(nptokens)
        for i in range(len(tokens)-sz+1):
            if nptokens==tokens[i:i+sz]:
                tokens[i:i+sz]=[phrase]
                break
    ##converting all tokens ex
    tagged_tokens=nltk.pos_tag(tokens)
    final_tokens=[]
    noun_types=['NN','NNP','NNS']
    for token,pos in tagged_tokens:
        if pos=='.':
            continue
        if not pos in noun_types:
            final_tokens.append(token.lower())
        else:
            final_tokens.append(token)
    return final_tokens
def fe3000(document):
    feats={}
    
    tokens=npcollocation(document)
    for pw in tokens:
        feats["has_phrase({})".format(pw)]=True
    return feats
def hashfun(num):
    m=hashlib.md5()
    m.update(str(random.random()))
    m.update(str(num))
    return m.hexdigest()
class ProBayesClassifier(NaiveBayesClassifier):
    def __init__(self):
        NaiveBayesClassifier.__init__(self,[],feature_extractor=fe3000)
        self.ref={}
        self.threshhold=0.06
    def threshholdratio(self):
        return 1.20
    def update_store(self,doc_train):
        key=hashfun(random.random())
        self.update([(doc_train,key)])
        self.ref.setdefault(key,[]).append(doc_train)
    def update_store_key(self,doc_train,key):
        self.update([(doc_train,key)])
        self.ref.setdefault(key,[]).append(doc_train)
    def outPerformAlgo(self,doc_test):
        prob_dist=self.prob_classify(doc_test)
        toppers=[]
        for key in self.ref.keys():
            if prob_dist.prob(key)>self.threshhold:
                toppers.append(prob_dist.prob(key))
            toppers.sort()
        if len(toppers)>1 and toppers[0]>2/len(toppers):
            return True
        
        if len(toppers)>2:
            z=0
            for tp in toppers:
                z+=tp
            S=0
            for i in range(len(toppers)-1):
                cmp=1.0*(toppers[i]/toppers[i+1]-1)
                if cmp>self.threshholdratio():
                    c=0
                else:
                    c=1
                S+=((-1)**(i+c))*cmp*1.0
            if S>0:
                return True
            else:
                return False
        else:
            return False
