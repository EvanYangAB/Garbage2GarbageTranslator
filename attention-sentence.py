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
MAX_LENGTH = 30
SOS_token = 0
EOS_token = 1
SENT_RATIO = 5

class SentenceDatabase:
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def addTag(self, tag):
        if tag not in self.tag2index:
            self.tag2index[tag] = self.n_tags
            self.index2tag[self.n_tags] = tag
            self.n_tags += 1

    # size limit for debuging purposes
    def __init__(self, split, size_limit = 10):   
        self.word2index = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        # self.tag2index = {}
        # self.index2tag = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
        self.n_tags = 2
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
            for q_sent, q_sent_evis in zip(d["sentences"], d["sent_evidences"]):
                self.sentence_pair.append([d["answer"].split(" [")[0], pos_tag(word_tokenize(q_sent)), [pos_tag(word_tokenize(sent_evi["sent_text"])) for sent_evi in q_sent_evis]])
            if size >= size_limit:
                break
        for answer, q_sent, evi_sentences in self.sentence_pair:
            for word in word_tokenize(answer):
                self.addWord(word)
            for word, tag in q_sent:
                self.addWord(word)
                # self.addTag(tag)
                self.addWord(tag)
            for sent in evi_sentences:
                for word, tag in sent:
                    self.addWord(word)
                    # self.addTag(tag)
                    self.addWord(tag)
        # filter sentences that is too long
        filtered_pairs = []
        for answer, sent, evids in self.sentence_pair:
            switch = len(sent) < MAX_LENGTH
            for e in evids:
                switch = switch and (len(e) < MAX_LENGTH)
            if switch:
                filtered_pairs.append([answer, sent, evids]) 
            print(switch)
        self.sentence_pair = filtered_pairs
 
s = SentenceDatabase('dev')
print(len(s.sentence_pair))
print(len(s.raw_questions))


def tensorFromSentence(lang, sentence):
    wordIndexes = [lang.word2index[i] for i in item for item in sentence]
    # tagIndexes = [lang.word2index[tag] for (word, tag) in sentence]
    wordIndexes.append(EOS_token)
    # tagIndexes.append(EOS_token)
    return torch.tensor(wordIndexes, dtype=torch.long, device=device).view(-1, 1), torch.tensor(tagIndexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromMatches(input, target):
    input_tensors = []
    for x in range(SENT_RATIO):
        input_tensors.append(tensorFromSentence(input_lang, input[x]))
    target_tensor = tensorFromSentence(output_lang, target)
    return (input_tensors, target_tensor)

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # this essensially gives how many dimensions a word is turned to
        self.word_embedding = nn.Embedding(input_size, hidden_size)
        # self.tag_embedding = nn.Embedding(tag_size, 5)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        # this flattens the embedding into 1 demension
        w_embedded = self.word_embedding(input).view(1, 1, -1)
        # t_embedded = self.tag_embedding(input[1]).view(1, 1, -1)
        # embedded = torch.cat((w_embedded, t_embedded), dim = 0)
        output = w_embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# outputs is of length max_length*sent_ratio, 
# hiddens is a sent_ratio length array with each tensor is hidden_size long, it is currently not used
# in this build
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length * SENT_RATIO)
        self.attn_combine = nn.Linear(self.hidden_size + self.max_length * SENT_RATIO, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # calculate the attention based on the previous word and the hidden
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # apply
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







teacher_forcing_ratio = 0.5

# persumebly, there is a 5 - 1 encoding ractio
def train(input_tensors, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_lengths = [input_tensor.size(0) for input_tensor in input_tensors]
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length*SENT_RATIO, encoder.hidden_size, device=device)
    encoder_hiddens = []

    loss = 0

    for sent_id in range(sent_ratio):
        for ei in range(input_lengths[sent_id]):
            encoder_output, encoder_hidden = encoder(
                input_tensors[sent_id][ei], encoder_hidden)
            encoder_outputs[sent_id*MAX_LENGTH+ei] = encoder_output[0, 0]
        encoder_hiddens.append(encoder_hidden)

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs, encoder_hiddens)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length