import os
import pickle
from pprint import pprint
from os.path import join as JP

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

from utils.nlp_utils import preproces
from utils.general import parse_yaml
from scripts.catalog import Catalog, load_catalog, load_corpus

import spacy
import sklearn
from spacy import displacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.catalog import (
    Catalog, Document, Corpus,
    load_catalog, load_corpus)

docker = os.environ.get('AM_I_IN_A_DOCKER_CONTAINER', False)
if docker: 
    print('I am inside a docker container')
    workdir = '/app/'
else: 
    print('I am running locally')
    workdir = 'c:\\Users\\RUIZP4\\Documents\\DOCS\\Pablo_Personal\\StanfordNLP\\Side_projects\\Document_Clustering'
    workdir = '/Users/pabloruizruiz/OneDrive/Courses/NLP_Stanford/Side_projects/Document_Clustering/'

tqdm.pandas()
os.chdir(workdir)
catalog = Catalog()
config = parse_yaml('config.yaml')
paths = config['paths']


# CONFIG
# ------

MODEL_NAME = 'TFIDF'
DOC_EMBED_SIZE = 10000 
CATALOG_NAME 'spacy_pipeline_on_EN_corpus'


nlp = spacy.load('en_core_web_sm')
def spacy_cleaning(
    document,
    tags_to_keep=['JJ', 'NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
    entities_to_remove=[]): #['ORG,NORP,GPE,PERSON']):
    ''' Keep some tags and remove some entities '''
    def pass_test(w, tags=tags_to_keep):
        if w.ent_type_ == 0:
                return w.tag_ in tags and not w.is_punct and not w.is_stop and w.ent_ not in entities_to_remove
        return w.tag_ in tags and not w.is_punct and not w.is_stop 
    words = [ word for word in document if pass_test(word)]
    tokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in words ]
    return ' '.join(tokens)  


''' DATA '''`
name = 'bbc-text.csv'
data = pd.read_csv(JP('data', name), index_col=0)

# for d,doc in enumerate(catalog.documents):
#     print('[INFO]: Parsing doc ',d)
#     catalog.documents[d].processed_text = spacy_cleaning(nlp(doc.clean_text))
# catalog.save(path=paths['catalog'],name='spacy_pipeline_on_EN_corpus')

catalog = Catalog()
documents = [Document() for i in range(data.shape[0])]
for d in range(len(documents)):
    documents[d].processed_text = data['processed'][d]
catalog.documents = documents


''' TFIDF '''
vectorizer = TfidfVectorizer(
    min_df=.05,
    max_df=.8,
    norm='l2',
    use_idf=True,
    smooth_idf=True,
    ngram_range=(1,3),
    lowercase=True,
    max_features=DOC_EMBED_SIZE,
    stop_words=stopwords.words('english'))

_ = catalog.collect_corpus(attr='processed_text', form=list)
tfidf = catalog.to_matrix(
    vectorizer=vectorizer,
    modelname=MODEL_NAME,
    max_docs=50)

tfidf.representation.head()

catalog.save(path=paths['catalog'],name=CATALOG_NAME)

