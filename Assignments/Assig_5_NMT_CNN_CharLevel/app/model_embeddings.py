#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
from cnn import CNN
from highway import Highway

CNN_KERNEL = 5
CHAR_EMBED = 50

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
        self.embeddings = nn.Embedding(len(vocab.id2char), CHAR_EMBED)  #, padding_idx=pad_token_idx)
        self.cnn = CNN(in_channels=CHAR_EMBED*21, out_channels=embed_size, kernel_size=CNN_KERNEL)
        self.highway = Highway()
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
        print('input: ', input.shape)
        input = input.permute(1,0,2)
        print('input: ', input.shape)
        x_emb = self.embeddings(input)                                  # (BS, sent_len, embbed, max_word_len)
        print('x_emb: ', x_emb.shape)       
        # TODO: Understand why this doesn't alter the result
        x_reshaped = x_emb.view(x_emb.shape[0],x_emb.shape[1], -1)      # (BS, sent_len, embbed * max_word)
        x_reshaped = x_reshaped.permute(0,2,1)                          # (BS, embbed * max_word, sent_len)
        print('x_reshaped: ', x_reshaped.shape)
        x_conv = self.cnn(x_reshaped)                   # (BS, embed_size)
        print('x_conv: ', x_conv.shape)
        x_high = self.highway(x_conv)                   # (BS, embed_size)
        print('x_high: ', x_high.shape)
        return self.dropout(x_high)