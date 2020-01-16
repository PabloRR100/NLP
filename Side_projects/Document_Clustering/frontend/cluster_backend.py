
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
from utils.general import parse_yaml, defaultdict_to_dict
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
# import hdbscan
from sklearn.decomposition import PCA
from scripts.algorithms.clustering import kmeans_clustering, kmedoids_clustering
from scripts.algorithms.clustering import plot_centroids_as_wordclouds


dim_reducters = [
    PCA(n_components=EMBED_SIZE),
    umap.UMAP(n_neighbors=5, n_components=EMBED_SIZE, metric='cosine')]

clustering_methods = [
    kmeans_clustering, 
    kmedoids_clustering]
    # hdbscan.HDBSCAN(min_samples=10, min_cluster_size=MIN_CLUSTER_SIZE)]

for reducter in dim_reducters:
    
    data_low_dim = reducter.fit_transform(data.copy())
    
    for k in range(MIN_K, MAX_K):
        
        for method in clustering_methods:
            
            print('[INFO]: Num_clusters = {} Method = {} Red_dim_technique = {}'.format(
                k, method.__name__, reducter.__class__.__name__))

            # Run algorithm
            clusters = method(data_low_dim.copy(), num_clusters=k)
            # words = words_per_cluster(tfidf, clusters)

            # Compute word importance -> centroids
            data['cluster'] = clusters.labels_
            centroids = data.groupby('cluster').mean().reset_index()
            scores = pd.melt(centroids, id_vars=['cluster'], var_name='word', value_name='score')
            
            # Collect data_low_dim with labes
            data_low_dim_df = pd.DataFrame(data_low_dim, columns=['d{}'.format(i+1) for i in range(data_low_dim.shape[1])])
            data_low_dim_df['cluster'] = clusters.labels_
            
            # Store clustering results
            # words_results[k][method][reducter] = scoress
            # embedding_results[k][method][reducter] = data_low_dim.copy()
            # words_results[k][method][reducter]['num_clusters'] = k 
            # embedding_results[k][method][reducter]['num_clusters'] = k
            cluster_results[k][method.__name__][reducter.__class__.__name__]['scatter'] = data_low_dim_df
            cluster_results[k][method.__name__][reducter.__class__.__name__]['wordcluds'] = scores

            # Create and store wordclouds
            # fig = plot_centroids_as_wordclouds(scores, n_cols=2-k%2)
            fig = plot_centroids_as_wordclouds(scores, n_cols=k)
            fig.savefig(JP(paths['images'], 'wordcloud_clusters_{}_{}_{}.png'.format(k, method.__name__, reducter.__class__.__name__)))

# SAVE
# name = '{}_{}_{}'.format(method.__name__, reducter.__class__.__name__, k)
print('[INFO]: Saving ...')
with open(JP(paths['results'], 'clustering.pkl'), 'wb') as f:
    pickle.dump(defaultdict_to_dict(cluster_results), f)

exit()
