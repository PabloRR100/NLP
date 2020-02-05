
import os
import spacy
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from os.path import join as JP

tqdm.pandas()

# spacy_model_path='globaldata/lang_models/english_sm'
spacy_model_path='en_core_web_sm'
nlp = spacy.load(spacy_model_path) # Powerfull model with everytihing included

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

name = '20newsgroup.csv'
path = JP('data', name)
data = pd.read_csv(path)

# Custom
data['len'] = data['text'].apply(lambda r: len(str(r)))
data = data[data['len'] > 3]

data['processed'] = data['text'].progress_apply(nlp).progress_apply(spacy_cleaning)
data.to_csv(JP('data', name+'-processed.csv'))

print('\nBefore Processing')
doc_list = [d for d in data.text]
print('Total documents: ', len(doc_list))
docs_toguether = ' '.join(d for d in doc_list)
all_words = docs_toguether.split(' ')
unique_words = np.unique(all_words)
print('All words: {}. Unique words: {}'.format(len(all_words), len(unique_words)))

print('\nAfter Processing')
doc_list = [d for d in data.processed]
print('Total documents: ', len(doc_list))
docs_toguether = ' '.join(d for d in doc_list)
all_words = docs_toguether.split(' ')
unique_words = np.unique(all_words)
print('All words: {}. Unique words: {}'.format(len(all_words), len(unique_words)))

print('Exiting...')
exit()