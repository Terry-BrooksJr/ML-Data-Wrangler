import re
import numpy as np
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from gensim.models.ldamodel import LdaModel
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from lda import LatentDirichletAllocator
from loguru import logger

nltk.download("stopwords")

class LatentDirichletAllocator:
    def __init__(self, corpus: str, num_of_topics:int=0) -> None:
       self.corpus = corpus 
       self.documents = ""
       self.id2word =  ""

    def create_document(self):
        vect = CountVectorizer()
        docs = vect.fit_transform(self.corpus)
        self.document = docs

    def preprocess_data(self):
        try:
            stop_words = stopwords.words("english")
            texts = [[
                word for word in simple_preprocess(str(doc)) if word not in stop_words
            ] for doc in self.documents]
            self.id2word = corpora.Dictionary(self.preprocess_data(texts))
        except Exception as e:
            logger.exception('Failed to Preprocess Data')
