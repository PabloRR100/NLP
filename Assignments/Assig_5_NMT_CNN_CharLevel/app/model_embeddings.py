#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn

from cnn import CNN
from highway import Highway

CNN_KERNEL = 5
CHAR_EMBED = 50
VERBOSE = False

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        # pad_token_idx = vocab.src['<pad>']
        """
        super(ModelEmbeddings, self).__init__()
        self.vocab = vocab
        self.embed_size = embed_size
        self.embeddings = nn.Embedding(len(vocab.id2char), CHAR_EMBED, padding_idx=vocab['<pad>'])
        self.cnn = CNN(in_channels=CHAR_EMBED, out_channels=embed_size, kernel_size=CNN_KERNEL)
        self.highway = Highway(embed_size=embed_size)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        # Move the BS first
        batch_size = list(input.shape)[1]
        if VERBOSE: print('input: ', input.shape)                   # (sent_len, BS, max_word_len)
        input = input.permute(1,0,2)                                # (BS, sent_len, max_word_len)
        if VERBOSE:  print('input: ', input.shape)
        input = input.reshape(-1, input.shape[-1])                  # (BS * sent_len, max_word)
        if VERBOSE:  print('input: ', input.shape)
        x_emb = self.embeddings(input)                              # (BS * sent_len, max_word_len, e_char)
        if VERBOSE:  print('x_emb: ', x_emb.shape)       
        x_reshaped = x_emb.permute(0,2,1)                           # (BS * sent_len embbed, e_char, max_word)
        if VERBOSE:  print('x_reshaped: ', x_reshaped.shape)
        x_conv = self.cnn(x_reshaped)                               # (BS * sent_len, e_char)
        # if VERBOSE:  print('x_conv: ', x_conv.shape)
        x_high = self.highway(x_conv)                               # (BS * sent_len, e_char)
        if VERBOSE:  print('x_high: ', x_high.shape)
        x_high = x_high.view(-1, batch_size, x_high.shape[-1])      # (sent_len, BS, e_char)
        if VERBOSE:  print('x_high: ', x_high.shape)
        return self.dropout(x_high)


