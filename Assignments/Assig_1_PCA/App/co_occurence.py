#!/usr/bin/env python

import os
import sys
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

sys.path.append(os.path.abspath(os.path.join('..')))

from utils.sanity_checks import *


def distinctWords(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): list of distinct words across the corpus, sorted (using python 'sorted' function)
            num_corpus_words (integer): number of distinct words across the corpus
    """
    corpus_words = []
    num_corpus_words = -1

    ### SOLUTION BEGIN
    corpus_words = set([word for document in corpus for word in document])
    num_corpus_words = len(corpus_words)
    corpus_words = sorted(list(corpus_words))
    ### SOLUTION END

    return corpus_words, num_corpus_words


def computeCoOccurrenceMatrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).
    
        Note: Each word in a document should be at the center of a window. Words near edges will have a smaller
              number of co-occurring words.
              
              For example, if we take the document "START All that glitters is not gold END" with window size of 4,
              "All" will co-occur with "START", "that", "glitters", "is", and "not".
    
        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (numpy matrix of shape (number of unique words in the corpus , number corpus words)): 
                Co-occurence matrix of word counts. 
                The ordering of the words in the rows/columns should be the same as the ordering of the words given by the distinct_words function.
            word2Ind (dict): dictionary that maps word to index (i.e. row/column number) for matrix M.
    """
    words, num_words = distinctWords(corpus)

    ### SOLUTION BEGIN
    M = np.zeros(shape=(num_words,num_words))
    word2Ind = {w:i for i,w in enumerate(words)}

    for document in corpus:
        for i, word in enumerate(document):
            context = document[max(0,i-window_size):i] + document[i+1:min(i+window_size+1,len(document))]
            for context_word in context:
                M[word2Ind[word], word2Ind[context_word]] += 1

    ### SOLUTION END

    # Nicer visualization
    # table = pd.DataFrame(M,index=words,columns=words)
    return M, word2Ind



def reduceToKDim(M, k=2):
    """ Reduce a co-occurence count matrix of dimensionality (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using the following SVD function from Scikit-Learn:
            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    
        Params:
            M (numpy matrix of shape (number of unique words in the corpus , number of number of corpus words)): co-occurence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (number of corpus words, k)): matrix of k-dimensioal word embeddings.
                    In terms of the SVD from math class, this actually returns U * S
    """
    np.random.seed(4355)
    n_iters = 10  # Use this parameter in your call to `TruncatedSVD`
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))

    ### SOLUTION BEGIN
    reducer = TruncatedSVD(n_components=k, n_iter=n_iters,)
    M_reduced = reducer.fit_transform(M)
    ### SOLUTION END

    print("Done.")
    return M_reduced



#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def test_distinctWords():
    print("\n\t\t\t Testing distinctWords \t\t\t")

    test_corpus = toyCorpus()
    test_corpus_words, num_corpus_words = distinctWords(test_corpus)

    ans_test_corpus_words = sorted(
        list(set(["START", "All", "ends", "that", "gold", "All's", "glitters", "isn't", "well", "END"])))
    ans_num_corpus_words = len(ans_test_corpus_words)

    print("\nYour Result:")
    print(
        "Words in corpus: {}\n Number of words in corpus: {}\n".format(test_corpus_words,
                                                                       num_corpus_words
                                                                       )
    )

    print("Expected Result:")
    print(
        "Words in corpus: {}\n Number of words in corpus: {}\n".format(ans_test_corpus_words,
                                                                       ans_num_corpus_words
                                                                       )
    )


def test_computeCoOccurenceMatrix():
    print("\n\t\t\t Testing computeCoOccurrenceMatrix \t\t\t")

    test_corpus = toyCorpus()
    M_test, word2Ind_test = computeCoOccurrenceMatrix(test_corpus, window_size=2)

    M_test_ans, word2Ind_test_ans = toyCorpusCoOccurence()

    for w1 in word2Ind_test_ans.keys():
        idx1 = word2Ind_test_ans[w1]
        for w2 in word2Ind_test_ans.keys():
            idx2 = word2Ind_test_ans[w2]
            student = M_test[idx1, idx2]
            correct = M_test_ans[idx1, idx2]
            if student != correct:
                print("Correct M:")
                print(M_test_ans)
                print("Your M: ")
                print(M_test)
                raise AssertionError(
                    "Incorrect count at index ({}, {})=({}, {}) in matrix M. Yours has {} but should have {}.".format(
                        idx1, idx2, w1, w2, student, correct))

    print("\nYour Result:")
    print(
        "Shape of co-occurence matrix: {}\n Word to index map: {}\n".format(M_test.shape,
                                                                            word2Ind_test
                                                                            )
    )

    print("\nExpected Result:")
    print(
        "Shape of co-occurence matrix: {}\n Word to index map: {}\n".format(M_test_ans.shape,
                                                                            word2Ind_test_ans
                                                                            )
    )


def test_reduceToKDim():
    print("\n\t\t\t Testing reduceToKDim \t\t\t")

    M_test_ans, word2Ind_test_ans = toyCorpusCoOccurence()
    M_test_reduced = reduceToKDim(M_test_ans, k=2)

    print("\nYour Result:")
    print(
        "Shape of reduced dim co-occurence matrix: {}\n".format(M_test_reduced.shape
                                                                )
    )

    print("\nExpected Result:")
    print(
        "Shape of reduced dim co-occurence matrix: {}\n".format((10, 2)
                                                                )
    )


if __name__ == "__main__":
    test_distinctWords()
    test_computeCoOccurenceMatrix()
    test_reduceToKDim()
