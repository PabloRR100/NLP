#!/usr/bin/env python

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10, 5]

from co_occurence import *

# Check Python Version
import sys
import os

assert sys.version_info[0] == 3
assert sys.version_info[1] >= 5

sys.path.append(os.path.abspath(os.path.join('..')))

from utils.utils import *


def plot_embeddings(M_reduced, word2Ind, words, title):
    for word in words:
        idx = word2Ind[word]
        x = M_reduced[idx, 0]
        y = M_reduced[idx, 1]
        plt.scatter(x, y, marker='x', color='red')
        plt.text(x, y, word, fontsize=9)
    plt.savefig(title)


# Read in the corpus
reuters_corpus = readCorpus()

M_co_occurrence, word2Ind_co_occurrence = computeCoOccurrenceMatrix(reuters_corpus)
M_reduced_co_occurrence = reduceToKDim(M_co_occurrence, k=2)
# Rescale (normalize) the rows to make them each of unit-length
M_lengths = np.linalg.norm(M_reduced_co_occurrence, axis=1)
M_normalized = M_reduced_co_occurrence / M_lengths[:, np.newaxis]  # broadcasting

words = ['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela']
plot_embeddings(M_normalized, word2Ind_co_occurrence, words, 'co_occurence_embeddings.png')
