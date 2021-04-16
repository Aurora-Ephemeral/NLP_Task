import string
from showprocess import ProgressBar
from nltk.corpus import stopwords
import WordFreq
from pygoogletranslation import Translator
import numpy as np
import pandas as pd
import nltk

class CleanText():

    def __init__(self,text):
        self.X = text

    def clean_text(self):
        progess_bar = ProgressBar(len(self.X))
        print('tokenization process:')
        for i,s in enumerate(self.X):
            progess_bar.update(i)
            s_no_punct=s.translate(str.maketrans('','',string.punctuation))
            s_token = s_no_punct.split()
            s_no_url = [token.lower() for token in s_token if (token.isalnum() and (not (token.startswith('https'))))]
            self.X[i] = s_no_url

        #text_svm_pipe = Pipeline([('count',StemmedCountVectorizer(stop_words='english',ngram_range=(1,3))),
                                  #('tf_idf',TfidfTransformer()),
                                 # ('clf',SGDClassifier(alpha=0.0005,n_jobs=-1))
                                 # ])

        clf_lang = WordFreq.WordFreq()
        #stopwords_eng = set(stopwords.words('english'))
        gs = Translator()
        #stopwords_ger = set(stopwords.words('german'))
        store_de_index = []
        store_de_text = []
        store_en_text = []
        store_en_index = []
        progess_bar = ProgressBar(len(self.X))
        print('cleaning process:')
        for i,s in enumerate(self.X):
            progess_bar.update(i)
            s_stemmed = []
            s_lang_type = clf_lang.guess_language(s)
            if s_lang_type == 'English':
                #s_stemmed = [stemmer_eng.stem(t) for t in s]
                s_no_sws = [t for t in s]
                s_en = [' '.join((s_no_sws))]
                store_en_text.extend(s_en)
                store_en_index.append(i)
            elif s_lang_type =='German_Deutsch':
                #s_no_sws = [stemmer_deu.stem(t) for t in s]

                #s_en = gs.translate(s_de[0], dest='en')
                #s_en2 = (s_en.text).split()
                s_no_sws = [t for t in s]
                s_de = [' '.join(s_no_sws)]
                store_de_text.extend(s_de)
                store_de_index.append(i)

            else:
                raise TypeError('Wrong type of language occurs, please check the data set')
            self.X[i] = [' '.join(s_no_sws)]

        translated_de = gs.translate(store_de_text,dest='en',src='de')

        for translations,idx in zip(translated_de,store_de_index):
            self.X[idx]=[translations.text]


        X_for_preprocess = []

        for s in self.X:
            X_for_preprocess.append(s[0])
        return X_for_preprocess