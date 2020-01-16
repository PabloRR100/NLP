
'''

Run all the clustering in a single runs
=======================================

1 - Load the Catalog created in main.py
2 - Apply different clustering algorithsm
3 - Store results for frontend visualization

'''

import os
import sys
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

DOCKER = os.environ.get('IM_IN_DOCKER', False)
if DOCKER: 
    WORKDIR = '/app/'
else:
    WORKDIR = 'c:\\Users\\RUIZP4\\Documents\\DOCS\\Pablo_Personal\\StanfordNLP\\Side_projects\\Document_Clustering'
    WORKDIR = '/Users/pabloruizruiz/OneDrive/Courses/NLP_Stanford/Side_projects/Document_Clustering/'

os.chdir(WORKDIR)
print('WORKING DIRECTORY: ', WORKDIR)
sys.path.append(WORKDIR)
sys.path.append(JP(WORKDIR,'utils'))
sys.path.append(JP(WORKDIR,'scripts'))
from utils.general import parse_yaml
from scripts.catalog import Catalog, load_catalog, load_corpus

# tqdm.pandas()
config = parse_yaml('config.yaml')
paths = config['paths']


''' 1 - Data ''' 

catalog = load_catalog(paths['catalog'], name='bbc_news')
tfidf = catalog.models['TFIDF']
data = tfidf.representation
# data = tfidf.representation.values


''' 2 - Clustering '''

MIN_K = 2
MAX_K = 4

EMBED_SIZE = 50
MIN_CLUSTER_SIZE = 50

words_results = defaultdict(lambda: defaultdict())
embedding_results = defaultdict(lambda: defaultdict())
nested_dict = lambda: defaultdict(nested_dict)
cluster_results = nested_dict()


import umap
import hdbscan
from sklearn.decomposition import PCA
from scripts.algorithms.clustering import kmeans_clustering, kmedoids_clustering
# from scripts.algorithms.clustering import words_per_cluster


dim_reducters = [
    PCA(n_components=EMBED_SIZE), 
    umap.UMAP(n_neighbors=5, n_components=EMBED_SIZE, metric='cosine')]

clustering_methods = [
    kmeans_clustering] 
    # kmedoids_clustering]
    # hdbscan.HDBSCAN(min_samples=10, min_cluster_size=MIN_CLUSTER_SIZE)]

for reducter in dim_reducters:
    for method in clustering_methods:
        for k in range(MIN_K, MAX_K):
            
            print('[INFO]: Num_clusters = {} Method = {} Red_dim_technique = {}'.format(
                k, method.__name__, reducter.__class__))

            # Run algorithm
            data_low_dim = reducter.fit_transform(data.copy())
            clusters = method(data_low_dim, njobs=-1, num_clusters=k)
            # words = words_per_cluster(tfidf, clusters)

            # Compute word importance -> centroids
            data['cluster'] = clusters.labels_
            centroids = data.groupby('cluster').mean().reset_index()
            scores = pd.melt(centroids, id_vars=['cluster'], var_name='word', value_name='score')
            
            # Collect data_low_dim with labes
            data_low_dim = pd.DataFrame(data_low_dim, columns=['d{}'.format(i+1) for i in range(data_low_dim.shape[1])])
            data_low_dim['cluster'] = clusters.labels_
            
            # Store
            # words_results[k][method][reducter] = scoress
            # embedding_results[k][method][reducter] = data_low_dim.copy()
            cluster_results[k][method.__name__][reducter.__class__]['scatter'] = data_low_dim
            cluster_results[k][method.__name__][reducter.__class__]['wordcluds'] = scores
            # words_results[k][method][reducter]['num_clusters'] = k 
            # embedding_results[k][method][reducter]['num_clusters'] = k

# SAVE
print('[INFO]: Saving...')
name = '{}_{}_{}'.format(method.__name__, reducter.__class__, k)
with open(JP(paths['results'], name+'.pkl'), 'wb') as f:
    pickle.dump(dict(cluster_results), f)

exit()
