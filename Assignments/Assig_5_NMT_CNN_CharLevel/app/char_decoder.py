#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()
        self.target_vocab = target_vocab
        self.criterion = nn.CrossEntropyLoss(ignore_index=target_vocab.char2id['<pad>'], reduction='sum')
        self.charDecoder = nn.LSTM(input_size=char_embedding_size, hidden_size=hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(target_vocab.id2char))
        self.decoderCharEmb = nn.Embedding(len(target_vocab.id2char), char_embedding_size, padding_idx=target_vocab.char2id['<pad>']) # padding_idx=target_vocab['<pad>'])
    
    def forward(self, input, dec_hidden=None, verbose=0):
        """ Forward pass of character decoder.                       **** (length = sent_len * BS)   ||   batch = v) ****

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        # 1 Lookup the character embeddings                                         # (sent_len * BS, word_len)
        if verbose == 1: print('Input: ', input.shape)
        X = self.decoderCharEmb(input)                                              # (sent_len * BS, word_len, char_emb)
        if verbose == 1: print('Input embedded: ', X.shape)
        # 2 Pass to the LSTM the input embedding and decoder hidden state
        hidden, (last_hidden, last_cell) = self.charDecoder(X, dec_hidden)          # (sent_len * BS, word_len, hidden), (1, word_len, hidden)
        if verbose == 1: print('Hidden encoded: ', hidden.shape)
        # 3 Compute the scores
        s = self.char_output_projection(hidden)                                     # (sent_len * BS, word_len, Vchar)
        if verbose == 1: print('Scores: ', s.shape)
        return s, (last_hidden, last_cell)


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.                      **** (length = max_word_len)   ||   batch = number of words) ****

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch, for every character in the sequence.
        """
        target = char_sequence[1:].contiguous().view(-1)
        print('char_sequence: ', char_sequence)
        print('target: ', target)

        # Forward pass the char_sequence
        scores, _ = self.forward(char_sequence[:-1], dec_hidden)            # (sent_len * BS, word_len, V_char), (sent_len * BS, word_len, hidden)

        # Cross-Entropy Loss
        scores = scores.view(-1, len(self.target_vocab.char2id))
        print('scores view :', scores.shape)
        loss = self.criterion(scores, target)
        return loss
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).

    def decode_greedy(self, initialStates, device, max_length=21, verbose=0):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """
        BS = initialStates[0].shape[1]
        if verbose == 1: print('Batch size = ', BS)
        
        output_word = []
        decoded_words = ['{'] * BS
        current_char = [self.target_vocab.char2id['{']] * BS                                    # ()

        if verbose == 1: print('Current char [0]: {} :: {}'.format(current_char[0], self.target_vocab.id2char[current_char[0]]))
        current_char_tensor = torch.tensor(current_char, device=device)                         # (BS, )
    
        h_prev, c_prev = initialStates                                                          # ()

        # Up to max word leght predicted one csharacter:
        for j,t in enumerate(range(max_length)):
            
            # Forward pass the char decoder LSTM
            if j == 0 and verbose == 1: print('Current char tensor: ', current_char_tensor.shape)
            if j == 0 and verbose == 1: print('Current char tensor unsqueezed shape: ', current_char_tensor.unsqueeze(0).shape)   # ()
            _, (h_t,c_t) = self.forward(current_char_tensor.unsqueeze(0), (h_prev, c_prev))     # ()
            
            # Compute Scores
            if j == 0 and verbose == 1: print('Hidden shape: ', h_t.shape)                    # ()
            if j == 0 and verbose == 1: print('Hidden squeezed shape: ', h_t.squeeze(0).shape)                    # ()
            scores = self.char_output_projection(h_t.squeeze(0))                                # ()
            if j == 0 and verbose == 1: print('scores: ', scores)
            if j == 0 and verbose == 1: print('scores shape: ', scores.shape)
            
            # Compute Probabilities
            P = F.log_softmax(scores, dim=1)                                                    # ()
            if j == 0 and verbose == 1: print('probabilities: ', P)
            if j == 0 and verbose == 1: print('probabilities shape: ', P.shape)
            
            # Compute predicted character  
            current_char_tensor = torch.argmax(P, dim=1)                                               # ()
            if j == 0 and verbose == 1: print('predicted_char: ', current_char_tensor)
            if j == 0 and verbose == 1: print('predicted_char shape: ', current_char_tensor.shape)
            
            # For batches we don't stop in <END> token
            # Update every predicted output with its prediction
            for i in range(BS):
                decoded_words[i] += self.target_vocab.id2char[current_char_tensor[i].item()]
            
            # Update hidden states
            h_prev, c_prev = h_t, c_t

        if verbose == 1: print('Decoded words: ', decoded_words)
        for i in range(BS):
            # Remove start token
            decoded_words[i] = decoded_words[i][1:]
            # Truncate from where <END> was predicted
            decoded_words[i] = decoded_words[i].partition('}')[0]

        return decoded_words






        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
