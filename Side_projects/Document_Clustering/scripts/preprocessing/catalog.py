
# For Debuggin to work on Visual Studio Code
import os
import sys
import json
import base64
from os.path import join as JP
from os.path import abspath, pardir

local = '/Users/pabloruizruiz/OneDrive/Courses/NLP_Stanford/Side_projects/Document_Clustering/'
DOCKER = os.environ.get('IM_IN_DOCKER', False) 
WORKDIR = '/app/' if DOCKER else local
sys.path.append(WORKDIR)
sys.path.append(JP(WORKDIR,'utils'))
sys.path.append(JP(WORKDIR,'scripts'))
print('PATH: ', sys.path)
# -------------------------------------------


# General Libraries
import copy
import pickle
import pandas as pd
from glob import glob
from os.path import join as JP
from collections import defaultdict
from utils.nlp_utils import is_sentence, preproces
from utils.general import parse_yaml,ensure_directories


# Libraries for Models Creation
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


class Model:
    '''
    Attributes:
        - name -> identifier
        - mapping -> funcion (vectorizer) that transforms text to vect
        - representation -> vector representation of the catalog
        - catalog -> pointer to the catalog it was created from
        - id2token -> id -to-> token mapping
        - token2id -> token -to-> id mapping
        - vocabulary -> token -to-> id mapping as DataFrame
    '''
    def __init__(self, 
            name:str=None, 
            mapping=None,  #:Vectorizer
            representation:pd.DataFrame=None, 
            id2token:dict=None,
            token2id:dict=None,
            vocabulary:pd.DataFrame=None,
            catalog=None): #:Catalog
        self.name = name
        self.mapping = mapping
        self.representation = representation
        self.id2token = id2token,
        self.token2id = token2id,
        self.vocabulary = vocabulary 
        self.catalog = catalog


class Corpus:
    '''
    Collection of Documetns
    '''
    def __init__(self):
        self.documents = list()
        self.docs_by_topic = defaultdict(list)

    def save(self, path, name):
        if '.corpus' not in name:
            name += '.corpus'
        with open(JP(path, name), 'wb') as f:
            pickle.dump(self, f)


class Catalog:
    '''
    Attributes:
        - documents -> Dict of 'Topic': List of Docuemnts of that topic
        - corpus -> Collection of tokens formed from all of its documents

    Filter Catalog:
        filters = dict(
            topic = ['isocyanate'],
            country = ['CA', 'AU', 'BR', 'WO'],
            raw_text_len = 100
        )

        catalog_test = catalog.filter_catalog(filters)
        print(len(catalog.documents))        --> 6709
        print(len(catalog_test.documents))   --> 1304 of only those contries, topics and lenghts
    '''
    def __init__(self):
        self.documents = list()
        self.docs_by_topic = defaultdict(list)
        self.corpus = defaultdict(str)
        self.models = dict()
        #self.models = defaultdict(Model) --> Gives problems with Pickle

    def load_corpus(self, corpus): #:Corpus  
        self.documents = corpus.documents
        self.docs_by_topic = corpus.docs_by_topic
        return '[OK] Corpus loaded into Catalog'

    def filter_by_topic(self, topic):
        if topic not in self.docs_by_topic.keys():
            print('[ERROR]: Topic not in Catalog')
            return None
        else:
            return self.docs_by_topic

    def filter_catalog(self, filters:dict):
        ''' Returns the catalog filtered by a dict of keys to filter '''
        err = "[ERROR]: filters must be a dict of list {'filter': [values,to,filter]"
        assert isinstance(filters, dict), err

        new_catalog = Catalog()
        for d, document in enumerate(self.documents):            
            F = 0
            for _, (filt, vals) in enumerate(filters.items()):
     
                if filt not in dir(document):
                    F += 1
                    print(['ERROR: Document {} doesnt have method {} to be able to filter for it'.format(d,filt)])
                    continue    
                
                # Handle case where only 1 filter passed
                if isinstance(vals, str):
                    vals = [vals]

                if isinstance(vals, list):
                    if document.__getattribute__(filt) in vals:
                        F += 1

                if isinstance(vals, int):
                    if document.__getattribute__(filt) > vals:
                        F += 1

            # If document pass all filters
            if F == len(filters.keys()):
                new_catalog.documents.append(document)
                new_catalog.docs_by_topic[document.topic].append(document)
        return new_catalog

    def _collect_corpus_as_list(self, attr):
        ''' Return and Update Catalog.corpus as a list of strings '''
        # Check type of text to collect is an attribute of the document
        err = lambda a,d: '[ERROR]: Document {} has no attribute {}'.format(d,a)
        docs = []
        for d in self.documents:
            assert hasattr(d, attr), err
            docs.append(getattr(d,attr))
        return  docs

    def _collect_corpus_as_string(self, attr):
        ''' Return and Update Catalog.corpus as a unique string '''
        return '\n'.join(self._collect_corpus_as_list(self.documents, attr))

    def collect_corpus(self, attr:str='clean_text', form:type=list):
        ''' Return and Update Catalog.corpus with the desired form '''
        types = [list, str]
        err = lambda t: 'Invalid argument type {}. Not in in '.form(t,types)
        if issubclass(form,list):
            self.corpus = self._collect_corpus_as_list(attr)
        if issubclass(form,str):
            self.corpus = self._collect_corpus_as_string(attr)
        return self.corpus
                
    def to_matrix(
        self,
        vectorizer,
        modelname:str='NoName',
        max_docs=5):
        model = Model()
        model.name = modelname,
        model.mapping = vectorizer
        representation = model.mapping.fit_transform(self.corpus[:max_docs])
        model.dtm_sparse = representation # To keep the sparse representation - Used in PyLDAVis 
        model.representation = pd.DataFrame(
            data=representation.toarray()[:max_docs],
            columns=model.mapping.get_feature_names())
        model.token2id = model.mapping.vocabulary_
        model.id2token = {v:k for k,v in model.token2id.items()}
        model.vocabulary = pd.DataFrame(model.token2id, index=model.token2id.values())
        self.models[modelname] = model
        return model

    def save(self, path, name):
        if '.catalog' not in name:
            name += '.catalog'
        with open(JP(path, name), 'wb') as f:
            pickle.dump(self, f)


class Document:

    def __init__(self, 
        topic=None, label=None, 
        country=None, ident=None, code=None, path=None, text=None):
        
        self.topic = topic
        self.label = label
        self.country = country
        self.id = ident
        self.code = code
        self.path = path

        self.raw_text = text
        self.raw_text_len = None
        self.clean_text = None    
        self.clean_text_len = None    
        self.processed_text = None

    def _read_json(self, doc):
        return base64.b64decode(
            doc["text"]).decode('latin1').replace("\x00", "")

    def read_document(self):
        ''' Scan and decode a text document ''' 
        with open(self.path, 'r') as fp:
            self.raw_text = self._read_json(json.load(fp))
            self.raw_text_len = len(self.raw_text)
            return self.raw_text

    def parse_document(self, document:str):
        ''' Return a text document as a string of words after applying a logic
        to remove useless sentences '''
        sentences = list()
        for _,sentence in enumerate(document.split('\n')):
            if is_sentence(sentence):
                sentences.append(preproces(sentence))
        return sentences

    def clean_document(self):
        ''' Update and returns a clean string out of a raw text file give its path'''
        self.clean_text = '. '.join([sentence for sentence in self.parse_document(self.raw_text)])
        self.clean_text_len = len(self.clean_text)
        return self.clean_text


def load_catalog(path,name):
    if '.catalog' not in name:
            name += '.catalog'
    with open(JP(path, name), 'rb') as f:
            return pickle.load(f)


def load_corpus(path,name):
    if '.corpus' not in name:
            name += '.corpus'
    with open(JP(path, name), 'rb') as f:
            return pickle.load(f)



if __name__ == '__main__':

    def path_to_docs(data_path, topic, classes, max_docs=None, verbose=0):
        ''' Return the list of paths to the documents '''
        d = 1
        corpus = Corpus()
        print('[INFO]: Creating Catalog')
        print('-------------------------')
        for clas in classes:
            countries = os.listdir(JP(data_path, '_'.join([topic,clas])))
            for _,country in enumerate(countries):
                idxs = os.listdir(JP(data_path, '_'.join([topic,clas]),country))
                for idx in idxs:
                    cs = os.listdir(JP(data_path, '_'.join([topic,clas]),country,idx))
                    for c in cs:
                        files = glob(JP(data_path, '_'.join([topic,clas]),country,idx,c,'oc_docnorm_*.json'))
                        if max_docs is not None and d > max_docs:
                            return corpus
                        # If there is files
                        if len(files) > 0:
                            document = Document(
                                topic=topic, 
                                label=clas,
                                country=country, 
                                ident=idx, 
                                code=c,
                                path = files[0])
                            # Read the Json
                            document.read_document()
                            if verbose > 0: 
                                print('Reading document {}: -- Topic:{} -- Country: {} -- Lenght: {}'.format(
                                    d, document.topic, document.country, document.raw_text_len))
                            # Perform a first clearning
                            document.clean_document()
                            if verbose > 0: 
                                print('Parsing document {}: -- Topic:{} -- Country: {} -- Lenght: {}'.format(
                                    d, document.topic, document.country, document.raw_text_len))
                            corpus.documents.append(document)
                            corpus.docs_by_topic[document.topic].append(document)
                            d += 1
        return corpus


    config = parse_yaml('config.yaml')
    paths = config['paths']
    ensure_directories(paths)

    VERBOSE = 1
    DATA_PATH = paths['data']
    TOPIC = config['topic']
    CLASSES = config['classes']

    corpus = path_to_docs(DATA_PATH, TOPIC, CLASSES, max_docs=None, verbose=VERBOSE)
    corpus.save(path=paths['catalog'],name='corpus1')
