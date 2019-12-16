import os
import pickle
from pprint import pprint
from os.path import join as JP

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import string

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

tqdm.pandas()
catalog = Catalog()
config = parse_yaml('config.yaml')
paths = config['paths']
os.chdir('c:\\Users\\RUIZP4\\Documents\\DOCS\\Pablo_Personal\\StanfordNLP\\Side_projects\\Document_Clustering')

nlp = spacy.load('en_core_web_sm') # Powerfull model with everytihing included
def spacy_cleaning(
    document,
    tags_to_keep=['JJ', 'NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
    entities_to_remove=['ORG,NORP,GPE,PERSON']):

    def pass_test(w, tags=tags_to_keep):
        if w.ent_type_ == 0:
                return w.tag_ in tags and not w.is_punct and not w.is_stop and w.ent_ not in entities_to_remove
        return w.tag_ in tags and not w.is_punct and not w.is_stop 

    words = [ word for word in document if pass_test(word)]
    tokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in words ]
    return ' '.join(tokens)  


''' DATA '''
name = '20newsgroup.csv'
data = pd.read_csv(JP('data',name)).iloc[:,1:]
data.head()

documents = [Document() for i in range(data.shape[0])]
for d in range(len(documents)):
    documents[d].processed_text = data['processed'][d]

catalog = Catalog()
catalog.documents = documents


''' TFIDF '''
EMBED_SIZE = 10000 
NUM_CLUSTERS = data['category'].nunique()
WORDS_PER_CLUSTER = None
print(NUM_CLUSTERS)

vectorizer = TfidfVectorizer(
    min_df=.05,
    max_df=.8,
    norm='l2',
    use_idf=True,
    smooth_idf=True,
    max_features=EMBED_SIZE,
    ngram_range=(1,3),
    lowercase=True,
    stop_words=stopwords.words('english'))

_ = catalog.collect_corpus(attr='processed_text', form=list)
tfidf = catalog.to_matrix(
    vectorizer=vectorizer,
    modelname='TFIDF',
    max_docs=50)

tfidf.representation.head()


''' LDA '''
NUM_TOPICS = NUM_CLUSTERS = 5

# Functions for printing keywords for each topic
def selected_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([vectorizer.get_feature_names()[i] for i in topic.argsort()[:-top_n - 1:-1]]) 


from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(
    n_components=NUM_TOPICS, 
    max_iter=100, 
    learning_method='online',
    verbose=True,)

# LOAD
with open(JP(paths['checkpoints'], 'lda_sklearn_local.pkl'), 'rb') as obj:
    lda = pickle.load(obj)
lda

# TRAIN
print('Latent Semantion Allocation')
data_lda = lda.fit_transform(tfidf.representation)

# Save
with open(JP(paths['checkpoints'], 'lda_sklearn_local.pkl'), 'wb') as obj:
    pickle.dump(lda,obj)

# Evaluation
# Keywords for topics clustered by Latent Dirichlet Allocation
print("LDA Model:")
selected_topics(lda, vectorizer, 8)

# Visualziation
import pyLDAvis
import pyLDAvis.sklearn
pyLDAvis.sklearn.prepare(lda, tfidf.representation.values, tfidf.dtm_sparse, mds='tsne')

