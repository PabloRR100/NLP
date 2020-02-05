
'''
Run all the clustering in a single runs
=======================================
'''

import os
import sys
import pickle
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from os.path import join as JP
from collections import defaultdict

''' System Configuration '''
local = 'c:\\Users\\RUIZP4\\Documents\\DOCS\\Pablo_Personal\\StanfordNLP\\Side_projects\\Document_Clustering'
local = '/Users/pabloruizruiz/OneDrive/Courses/NLP_Stanford/Side_projects/Document_Clustering/'
DOCKER = os.environ.get('IM_IN_DOCKER', False) 
DOCKER = True # TODO: Hardcoded now because image is broken and it doesn't have this ENV variable yet
WORKDIR = '/app/' if DOCKER else local

os.chdir(WORKDIR)
print('WORKING DIRECTORY: ', WORKDIR)
sys.path.append(WORKDIR)
sys.path.append(JP(WORKDIR,'utils'))
sys.path.append(JP(WORKDIR,'scripts'))

import umap
# import hdbscan
from sklearn.decomposition import PCA
from scripts.preprocessing.catalog import load_catalog
from utils.general import parse_yaml, defaultdict_to_dict
from scripts.algorithms.clustering import kmeans_clustering, kmedoids_clustering

''' 0 - Conf '''

config_file = parse_yaml('config.yaml')  # A config map
config = parse_yaml(config_file)
paths = config['paths']


''' 1 - Data ''' 

# TODO: Data is going to be passed !? --> Simulated with arg parse
import argparse
parser = argparse.ArgumentParser(description='Introduce Parameters From External Call')
parser.add_argument('--catalog', '-c', default='bbc_news', help='Catalog to load data from')
args = parser.parse_args()

''' 2 - Clustering '''

print('Loading configuration parameters...')

# General
MIN_K = config['parameters']['general']['MIN_K']
MAX_K = config['parameters']['general']['MAX_K']
EMBED_SIZE = config['parameters']['general']['EMBED_SIZE']

# Umap Specific
METRIC = config['parameters']['umap']['metric']
N_NEIGHS = config['parameters']['umap']['n_neighbors']
UMAP_EPOCHS = config['parameters']['umap']['epochs']
N_COMPONENTS = config['parameters']['umap']['n_components']
MIN_CLUSTER_SIZE = config['parameters']['umap']['min_cluster_size']

dim_reducters = [
    PCA(n_components=EMBED_SIZE, random_state=46),
    umap.UMAP(
        n_epochs=UMAP_EPOCHS, n_neighbors=N_NEIGHS, 
        n_components=N_COMPONENTS, metric=METRIC, 
        random_state=46, verbose=True)]

clustering_methods = [
    kmeans_clustering, 
    kmedoids_clustering]
    # hdbscan.HDBSCAN(min_samples=10, min_cluster_size=MIN_CLUSTER_SIZE)]

def cluster(dim_reducters:list, MIN_K:int, MAX_K:int, catalog_name:str=None):

    global args

    # Data
    if catalog_name is None:
        catalog_name = args['catalog']
    catalog = load_catalog(paths['catalog'], name=catalog_name)
    tfidf = catalog.models['TFIDF']
    data = tfidf.representation

    # Objects to store the results
    nested_dict = lambda: defaultdict(nested_dict)
    cluster_results = nested_dict()

    for reducter in tqdm(dim_reducters):
        
        data_low_dim = reducter.fit_transform(data.copy())
        
        for k in range(MIN_K, MAX_K):
            
            for method in clustering_methods:
                
                print('[INFO]: Num_clusters = {} Method = {} Red_dim_technique = {}'.format(
                    k, method.__name__, reducter.__class__.__name__))

                # Run algorithm
                clusters = method(data_low_dim.copy(), num_clusters=k)
                
                # Compute word importance -> centroids
                data['cluster'] = clusters.labels_
                centroids = data.groupby('cluster').mean().reset_index()
                scores = pd.melt(centroids, id_vars=['cluster'], var_name='word', value_name='score')
                
                # Collect data_low_dim with labes
                data_low_dim_df = pd.DataFrame(data_low_dim, columns=['d{}'.format(i+1) for i in range(data_low_dim.shape[1])])
                data_low_dim_df['cluster'] = clusters.labels_
                
                # Store clustering results
                cluster_results[k][method.__name__][reducter.__class__.__name__]['scatter'] = data_low_dim_df
                cluster_results[k][method.__name__][reducter.__class__.__name__]['wordclouds'] = scores

    # SAVE
    print('[INFO]: Saving resutls ...')
    with open(
        JP(paths['results'], 'clustering_{}.pkl'.format(
            datetime.now().strftime("%Y-%m-%d__%H-%M"))), 'wb') as f:
        pickle.dump(defaultdict_to_dict(cluster_results), f)

    return 
    
# BACKUP CODE
# Create and store wordclouds
# fig = plot_centroids_as_wordclouds(scores, n_cols=1, figsize=(15,15))
# fig = plot_centroids_as_wordclouds(scores, n_cols=2-k%2)
# fig.savefig(JP(paths['images'], 'wordcloud_clusters_{}_{}_{}.png'.format(k, method.__name__, reducter.__class__.__name__)))
