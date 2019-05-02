from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import json
import os
import string
import re
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
MAX_LENGTH = 40

class SentenceDatabase:
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def addTag(self, tag):
        if tag not in self.tag2index:
            self.tag2index[word] = self.n_tags
            self.index2word[self.n_tags] = tag
            self.n_tags += 1

    def __init__(self, split, size_limit = 10):   
        self.word2index = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.tag2index = {}
        self.index2word = {}
        self.n_words = 2  # Count SOS and EOS
        self.n_tags = 0
        '''
        split can be {'train', 'dev', 'test'} - gets both the buzzer and guesser folds from the corresponding data file.
        '''
        dataset_path = os.path.join('qanta.'+split+'.json')
        evidence_path = os.path.join('qanta.'+split+'.evidence.text.json')
        with open(dataset_path) as fd, open(evidence_path) as fe:
            self.dataset = json.load(fd)
            self.evidences = json.load(fe)

        self.version = self.dataset['version']
        self.raw_questions = self.dataset['questions']
        self.data = {}
        for question in self.raw_questions:
            q_data = {key: question[key] for key in ["category", "subcategory", "text", "answer", "qanta_id"]}
            q_data["sentences"] = sent_tokenize(q_data["text"].replace(u'\xa0', u' '))
            q_id = q_data["qanta_id"]
            self.data[q_id] = q_data
        self.evidences = self.evidences["evidence"]
        for evidence in self.evidences:
            q_id = evidence["qanta_id"]
            self.data[q_id]["sent_evidences"] = evidence["sent_evidences"]

        # generate dictionary of words
        # create paired {answer, sentence, sent_evidences}
        self.sentence_pair = []
        size = 0
        for k, d in self.data.items():
            size += 1
            print(k)
            for q_sent, q_sent_evis in zip(d["sentences"], d["sent_evidences"]):
                self.sentence_pair.append([d["answer"].split(" [")[0], pos_tag(word_tokenize(q_sent)), [pos_tag(word_tokenize(sent_evi["sent_text"])) for sent_evi in q_sent_evis]])
            if size >= size_limit:
                break
        # for answer, q_sent, evi_sentences in self.sentence_pair:
        #     for word in word_tokenize(answer):
        #         self.addWord(word)
        #     for word, tag in word_tokenize(q_sent):
        #         self.addWord(word)
        #         self.addTag(tag)
        #     for sent in evi_sentences:
        #         for word, tag in word_tokenize(sent):
        #             self.addWord(word)
        #             self.addTag(tag)
 
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # this essensially gives how many dimensions a word is turned to
        self.word_embedding = nn.Embedding(input_size, hidden_size)
        self.tag_embedding = nn.Embedding(input_size, 10)
        self.gru = nn.GRU(hidden_size + 10, hidden_size)

    def forward(self, input, hidden):
        # this flattens the embedding into 1 demension
        w_embedded = self.word_embedding(input).view(1, 1, -1)
        t_embedded = self.tag_embedding(input).view(1, 1, -1)
        embedded = torch.cat((w_embedded, t_embedded), dim = 0)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



s = SentenceDatabase("dev")
print(s.raw_questions[0])