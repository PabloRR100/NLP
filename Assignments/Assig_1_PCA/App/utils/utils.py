#!/usr/bin/env python

import nltk

nltk.download('reuters')
from nltk.corpus import reuters

START_TOKEN = '<START>'
END_TOKEN = '<END>'


def readCorpus(category="crude"):
    """ Read files from the specified Reuter's category.
        Params:
            category (string): category name
        Return:
            list of lists, with words from each of the processed files
    """
    files = reuters.fileids(category)
    return [[START_TOKEN] + [w.lower() for w in list(reuters.words(f))] + [END_TOKEN] for f in files]
