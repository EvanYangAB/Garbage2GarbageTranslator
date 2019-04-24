from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import json
import os
import string
import re
import random
from nltk.tokenize import sent_tokenize, word_tokenize
# import torch
# import torch.nn as nn
# from torch import optim
# import torch.nn.functional as F
    

class SentenceDatabase:
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def __init__(self, split):   
        self.word2index = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
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
            q_data["sentences"] = sent_tokenize(q_data["text"])
            q_id = q_data["qanta_id"]
            self.data[q_id] = q_data
        self.evidences = self.evidences["evidence"]
        for evidence in self.evidences:
            q_id = evidence["qanta_id"]
            self.data[q_id]["sent_evidences"] = evidence["sent_evidences"]

        # generate dictionary of words
        # create paired {answer, sentence, sent_evidences}
        self.sentence_pair = []
        for _, d in self.data.items():
            for q_sent, q_sent_evis in zip(d["sentences"], d["sent_evidences"]):
                self.sentence_pair.append([d["answer"].split(" [")[0], q_sent, [sent_evi["sent_text"] for sent_evi in q_sent_evis]])
        for answer, q_sent, evi_sentences in self.sentence_pair:
            for word in word_tokenize(answer):
                self.addWord(word)
            for word in word_tokenize(q_sent):
                self.addWord(word)
            for sent in evi_sentences:
                for word in word_tokenize(sent):
                    self.addWord(word)
    
s = SentenceDatabase("dev")
print(s.sentence_pair[3])
print(s.word2index[word_tokenize(s.sentence_pair[3][0])[0]])
print(s.word2index[word_tokenize(s.sentence_pair[3][0])[1]])
print(s.word2index[word_tokenize(s.sentence_pair[3][0])[2]])
print(s.word2index[word_tokenize(s.sentence_pair[3][0])[3]])