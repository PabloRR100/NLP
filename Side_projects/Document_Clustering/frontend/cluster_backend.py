
import os
import pickle
from os.path import join as JP

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

import spacy
import sklearn
from spacy import displacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# os.chdir('/app/')
os.chdir('/Users/pabloruizruiz/OneDrive/Courses/NLP_Stanford/Side_projects/Document_Clustering/')
print('WORKING DIRECTORY: ', os.getcwd())
sys.path.append(os.getcwd())
sys.path.append(JP(os.getcwd(),'utils'))
sys.path.append(JP(os.getcwd(),'scripts'))
from utils.general import parse_yaml
from scripts.catalog import Catalog, load_catalog, load_corpus

# tqdm.pandas()
config = parse_yaml('config.yaml')
paths = config['paths']

''' 1 - Data ''' 

# Import 
# data = pd.read_csv(JP(paths['data'],'bbc-text-processed.csv'), index_col=0)
catalog = load_catalog(paths['catalog'], name='bbc')
print(type(catalog.models))

# TFIDF
tfidf = catalog.models['TFIDF']
print(tfidf.representation.shape)
tfidf.representation.head()

# Clustering
from scripts.algorithms.clustering import *

MIN_K = 2
MAX_K = 10
words_results = defaultdict(lambda: defaultdict())
cluster_results = defaultdict(lambda: defaultdict())
embedding_results = defaultdict(lambda: defaultdict())

