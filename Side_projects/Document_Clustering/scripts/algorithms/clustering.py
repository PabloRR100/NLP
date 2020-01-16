

import numpy as np
import pandas as pd
from copy import deepcopy
from collections import defaultdict

from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

import requests
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud


'''
UTILS
=====
'''

def filter_cluster(model, clusters, i):
    ''' Returns the tfidf representation for the centroid of a cluster '''
    return pd.DataFrame(
        clusters.cluster_centers_[i].reshape(1,-1), 
        columns=model.representation.columns)

def compute_word_importance_for_centroid(cluster,words_per_cluster):
    ''' Return the first (words_per_cluster) with the highest TFIDF score
    for the centroid of that cluster as a cluster summary document '''
    scores = pd.DataFrame(cluster.T.reset_index())
    scores.columns=['word','centroid_score']
    return scores.sort_values(by='centroid_score',ascending=False)[:words_per_cluster]


def subsample_tfidf_by_cluster(tfidf:pd.DataFrame, clusters:KMeans, cluster_id:int):
    ''' Returns a sample of the TFIDF keeping only the documents (rows) that belong to a particular cluster '''
    return tfidf.loc[clusters.labels_==cluster_id]

def subsample_tfidf_by_terms(tfidf:pd.DataFrame, words):
    ''' Return a sample version of a TFIDF for the words of a cluster 
    NOTE: This function implmented compute_word_importance()'''
    return tfidf[words]

def cluster_to_tfidf(model, clusters:KMeans, cluster_id:int):
    submodel = deepcopy(model)
    submodel.representation = subsample_tfidf_by_cluster(model.representation, clusters, cluster_id)
    return submodel


def get_tf_idf_of_word_from_tfidf_matrix(model,k,v):
    ''' Get the TF vector and IDF float of an specific term '''
    return model.representation[[k]].values, model.mapping.idf_[v]

def compute_word_importance_using_documents(model, words_of_cluster=None):
    ''' Compute the importance of a word given the TFIDF for a bunch of
    importance methods '''
    scores = defaultdict(list)
    if not words_of_cluster: words_of_cluster = model.token2id.keys()
    print('[INFO]: Computing word importance for each cluster')
    for k,v in model.token2id.items():
        # With this comparison we save a lot of time
        if k in words_of_cluster:
            scores['word'].append(k)
            t,i = get_tf_idf_of_word_from_tfidf_matrix(model,k,v)
            scores['idf'].append(i)
            scores['max_tf_idf'].append(np.max(t)) # *i
            scores['avg_tf_idf'].append(np.mean(t)) # *i
            scores['norm_tf_idf'].append(np.linalg.norm(t)) # *i
    # scores['centroid_score'] = scores_for_centroid()
    return scores


'''
REGULAR CLUSTERING 
==================
'''

''' ALGORITHMS '''
def kmean_clustering(
    data:pd.DataFrame,
    num_clusters:int=4,
    njobs=-1,
    random_state=46):
    '''
    Perform K-Means Algorithm
    Args:
        - model: Trained instance of class Model
        - num_clusters: Number of Clusters to look for

    Returns: 
        - Clustering model instance
    '''
    km = KMeans(
        n_clusters=num_clusters,
        init='k-means++',
        n_init=20,
        max_iter=1000,
        n_jobs=njobs,
        random_state=random_state)
    return km.fit(data)

def kmedoids_clustering(
    model, # class Model
    num_clusters:int=4,
    metric='cosine',
    random_state=46):
    km = KMedoids(
        n_clusters=num_clusters,
        metric=metric,
        random_state=random_state)
    return km.fit(model.representation)


''' PLOTS  '''

def mask_(url="https://image.shutterstock.com/image-illustration/flask-word-cloud-artwork-isolated-260nw-185529119.jpg"):
    return np.array(
        Image.open(
            requests.get(url,stream=True).raw))

def define_subplots(n_cols,n_plots,figsize=None):
    '''Return the axes given a total 
    of plots and desired number of columns'''
    j = 1 if n_plots%n_cols != 0 else 0
    n_rows = (n_plots // n_cols) + j 
    
    if not figsize: 
        figsize=(n_cols*5,n_rows*5)
    
    fig, axs = plt.subplots(
        nrows=n_rows, ncols=n_cols, sharex=False, sharey=False,
        figsize=figsize)
    
    return fig,axs

def cluster_to_wordcloud(
    df, max_words=200, use_mask=False, bgcolor='black'):
    ''' Convert 1 cluster into a WordCloud given:
        - The TFIDF for the cluster
        - The Score Method that give imporance to the word '''
    # Create the wordcloud attending to the inverse of idf
    wordcloud = WordCloud(
        max_words=max_words, 
        mask=mask_ if use_mask else None,
        background_color=bgcolor).generate_from_frequencies(
            frequencies=dict(zip(df.word, df.score)))
    return wordcloud

def plot_centroids_as_wordclouds(
    word_scores,
    NUM_CLUSTERS = None,
    max_words_per_cloud=100, 
    use_mask=False, n_cols=2, figsize=(15,15), show=True):
    ''' Convert a centroid representation to its correspondent wordcloud '''
    if not NUM_CLUSTERS:
        NUM_CLUSTERS = word_scores.cluster.nunique()

    n_plots = NUM_CLUSTERS
    fig, axs = define_subplots(n_cols,n_plots, figsize)
    
    for c in range(NUM_CLUSTERS):
        wordcloud = cluster_to_wordcloud(
            df=word_scores[word_scores.cluster == c],
            max_words=max_words_per_cloud,
            use_mask=use_mask)
        
        # Plot the resulting wordcloud
        if len(axs.shape) == 1:
            axs[c].imshow(wordcloud)
            axs[c].axis('off')            
        else:
            axs[c // n_cols, c % n_cols].imshow(wordcloud)
            axs[c // n_cols, c % n_cols].axis('off')
    plt.tight_layout()
    if not show:
        return fig
    else:
        plt.show()
    return


def plot_clusters_as_wordclouds(
    tfidf:pd.DataFrame, 
    cluster_words:dict,
    method:str='idf',
    max_words_per_cloud=100, use_mask=False, n_cols=2):
    '''
    Arguments:
        - tfidf: TFIDF of the entire Corpus
        - cluster_words: Dict {'cluster_id': [list of important words]}
        - methods: the Score Method that give imporance to the word in that cluster
    Steps:
        - Iterate for each cluster
        - Subsample the TFIDF to the Cluster TDIDF (reduce the columns to increase performance)
        - Get the scores of the chosen methods to give importance to the words --> Not needed ???
        - Call cluster_to_wordcloud() for that cluster to get its corresponding wordcloud
    '''
    n_rows = len(cluster_words)//n_cols
    _, axs = plt.subplots(nrows=n_rows, ncols=n_cols,figsize=(n_cols*5,n_rows*5))
    
    for cluster,words in cluster_words.items():
        
        cluster_word_scores = pd.DataFrame(compute_word_importance_using_documents(tfidf,words))
        wordcloud = cluster_to_wordcloud(
            df=cluster_word_scores,
            method='norm_tf_idf',
            max_words=max_words_per_cloud,
            use_mask=use_mask)
        
        # Plot the resulting wordcloud
        axs[cluster // n_cols, cluster % n_cols].imshow(wordcloud)
        axs[cluster // n_cols, cluster % n_cols].axis('off')
    plt.tight_layout()
    plt.show()
    return



def plot_subsampled_clusters_as_wordclouds(
    tfidf:pd.DataFrame, 
    clusters:KMeans,
    cluster_words:dict,
    method:str='idf',
    max_words_per_cloud=100, use_mask=False, n_cols=2):
    '''
    Arguments:
        - tfidf: TFIDF of the entire Corpus
        - cluster_words: Dict {'cluster_id': [list of important words]}
        - methods: the Score Method that give imporance to the word in that cluster
    Steps:
        - Iterate for each cluster
        - Subsample the TFIDF to the Cluster TDIDF (reduce the columns to increase performance)
        - Get the scores of the chosen methods to give importance to the words --> Not needed ???
        - Call cluster_to_wordcloud() for that cluster to get its corresponding wordcloud
    '''
    n_rows = len(cluster_words)//n_cols
    _, axs = plt.subplots(nrows=n_rows, ncols=n_cols,figsize=(n_cols*5,n_rows*5))
    
    for cluster,words in cluster_words.items():
        
        subtfidf = cluster_to_tfidf(tfidf,clusters,cluster)
        cluster_word_scores = pd.DataFrame(compute_word_importance_using_documents(subtfidf,words))

        wordcloud = cluster_to_wordcloud(
            df=cluster_word_scores,
            method='norm_tf_idf',
            max_words=max_words_per_cloud,
            use_mask=use_mask)
        
        # Plot the resulting wordcloud
        axs[cluster // n_cols, cluster % n_cols].imshow(wordcloud)
        axs[cluster // n_cols, cluster % n_cols].axis('off')
    plt.tight_layout()
    plt.show()
    return




''' ================================================================================================================ '''




'''
HIERARCHICAL CLUSTERING 
=======================
'''

# def plot_dendogram_from_catalog(
#     model, 
#     n_terms:int,
#     truncate_mode=None,
#     clusters:int=5):
#     linkage_matrix = ward_clustering(model=model,n_terms=n_terms)
#     terms = get_most_relevant_terms(tfidf_df=model,n_terms=n_terms)
#     plot_dendogram_from_linkage_matrix(
#         linkage_matrix=linkage_matrix,
#         truncate_mode=truncate_mode,
#         clusters=clusters,
#         labels=terms)
#     return


def compute_dist_matrix(df,metric='cosine'):
    return pdist(df, metric=metric)


def hca_document_clustering(
    model, # Model
    method:str='ward',
    distance_metric:str='cosine'):
    '''
    Performs Hierarchical Cluster
    Arguments:
        - model: Model instance representation
        - methods: How the distance between clusters is minimized
        - distance_metric: How to measure the distance between 2 clusters
    NOTE: 
        Filtering by terms is breaking the whole thing when plotting ??
        Does it make sense to run it on all and then truncate the plot rather than
        subsampling the TFIDF by the most important words (columns)?
    '''
    X = model.representation
    print('[INFO]: Computing Distance Matrix using {} distance'.format(distance_metric))
    d = compute_dist_matrix(X,distance_metric)
    print('[INFO]: Performing Hierarchical Clustering using {} linkage'.format(method))
    linkage_matrix = linkage(y=d, method=method)
    return linkage_matrix


def plot_dendogram_from_linkage_matrix(
    linkage_matrix, 
    truncate_mode:str=None,
    p:int=None,
    labels:list=None,
    orientation='right',
    show_leaf_counts=True,
    leaf_rotation=0.,
    leaf_font_size=12,
    figsize=None):
    ''' Plot a dendogram out of its linkage matrix '''
    
    _, ax = plt.subplots(figsize=figsize) 

    dendrogram(
        Z=linkage_matrix,
        p=p,                            
        truncate_mode=truncate_mode,    
        orientation=orientation, 
        show_leaf_counts=show_leaf_counts,
        leaf_rotation=leaf_rotation,
        leaf_font_size=leaf_font_size,
        labels=labels, 
        ax=ax)

    plt.tick_params(
        axis= 'x',          
        which='both',      
        bottom='off',     
        top='off',       
        labelbottom='off')

    plt.tight_layout() #show plot with tight layout
    plt.show()
    return


''' RETRIEVE INFORMATION AFTER THE CLUSTERING'''

def retrieve_doc_idx_by_level(cluster_idx, idx):
    ''' Return the indexes of the documents that match the cluster id '''
    doc_ids = np.array(list(range(len(cluster_idx))))
    mask = cluster_idx==idx
    return [c for c,i in zip(doc_ids,mask) if i]


def retrieve_hca_info(Z, criterion='maxclust', min_clusters=2, max_clusters=None):
    if not max_clusters: max_clusters = len(Z)//2+1
    ''' Retrive the documents that belong to each cluster 
    for every merge done during the HCA process ''' 
    cluster_dict = defaultdict(lambda: defaultdict(list))
    for level in range(min_clusters, max_clusters):
        cluster_idx = fcluster(Z, level, criterion=criterion)
        for c in range(level):
            cluster_dict[level][c].append(retrieve_doc_idx_by_level(cluster_idx,c+1))
    return cluster_dict


if __name__ == '__main__':

    '''
    PIPELINE UNTIL THE TFIDF
    ------------------------
    '''    
    import sys
    sys.path.append('../../utils')
    sys.path.append('../../scripts')
    from nltk.corpus import stopwords
    from utils.general import parse_yaml
    from scripts.catalog import load_catalog, Catalog
    from sklearn.feature_extraction.text import TfidfVectorizer

    config = parse_yaml('config.yaml')
    paths = config['paths']

    catalog = Catalog()
    catalog = load_catalog(path=paths['catalog'], name='spacy_pipeline_on_US_corpus')
    catalog.collect_corpus(attr='processed_text', form=list)

    ''' TFIDF '''
    vectorizer = TfidfVectorizer(
        min_df=.1,
        max_df=.7,
        norm='l2',
        use_idf=True,
        smooth_idf=True,
        max_features=3000,
        ngram_range=(1,3),
        lowercase=True,
        stop_words=stopwords.words('english'))

    tfidf = catalog.to_matrix(
        vectorizer=vectorizer,
        modelname='TFIDF',
        max_docs=50)

    tfidf.representation.head()

    '''
    FLAT CLUSTERING
    ---------------
    '''
    NUM_CLUSTERS = 4
    EMBED_SIZE = 10000
    WORDS_PER_CLUSTER = 50

    clustered_words = kmeans_clustering(
        model=catalog.models['TFIDF'],
        num_clusters=NUM_CLUSTERS, 
        words_per_cluster=WORDS_PER_CLUSTER)

    ''' Clustering2WordCloud '''
    plot_clusters_as_wordclouds(tfidf, clustered_words, method='idf')



    # '''
    # HIERARCHICAL CLUSTERING
    # -----------------------
    # '''
    # # Alternative 1 - Computing everything in advanced
    # MAX_TERMS = 500
    # terms = tfidf_df.sort_values(
    #     by='idf',ascending=False).iloc[:MAX_TERMS,:]['word'].tolist()

    # X = catalog.models['TFIDF'].representation
    # X = X[terms]
    # X.head(5)

    # dist = 1 - cosine_similarity(X)
    # linkage_matrix = ward(dist)
    # plot_dendogram_from_linkage_matrix(linkage_matrix, terms)



    # # Aternative 2 - Directly from the catalog.models['TFIDF']
    # matrix = ward_clustering(
    #     model=catalog.models['TFIDF'],
    #     tfidf_df=tfidf_df,
    #     terms = terms)

    # ''' Clustering to Dendogram ''' 
    # plot_dendogram_from_linkage_matrix(matrix, clusters=5)
    # # plot_dendogram_from_linkage_matrix(catalog.models['TFIDF'])
