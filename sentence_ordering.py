import argparse
import nltk
import json
import pandas as pd
import numpy as np
import time
import os
from os import path
from typing import List, Dict, Iterable, Optional, Tuple, NamedTuple
from collections import defaultdict
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from scipy.sparse import coo_matrix, hstack

class Question(NamedTuple):
    qanta_id: int
    text: str
    first_sentence: str
    tokenizations: List[Tuple[int, int]]
    answer: str
    page: Optional[str]
    fold: str
    gameplay: bool
    category: Optional[str]
    subcategory: Optional[str]
    tournament: str
    difficulty: str
    year: int
    proto_id: Optional[int]
    qdb_id: Optional[int]
    dataset: str

    def to_json(self) -> str:
        return json.dumps(self._asdict())

    @classmethod
    def from_json(cls, json_text):
        return cls(**json.loads(json_text))

    @classmethod
    def from_dict(cls, dict_question):
        return cls(**dict_question)

    def to_dict(self) -> Dict:
        return self._asdict()

    @property
    def sentences(self) -> List[str]:
        return [self.text[start:end] for start, end in self.tokenizations]

    def runs(self, char_skip: int) -> Tuple[List[str], List[int]]:
        char_indices = list(range(char_skip, len(self.text) + char_skip, char_skip))
        return [self.text[:i] for i in char_indices], char_indices


class QantaDatabase:
    def __init__(self, split):
        '''
        split can be {'train', 'dev', 'test'} - gets both the buzzer and guesser folds from the corresponding data file.
        '''
        dataset_path = os.path.join('..', 'qanta.'+split+'.json')
        with open(dataset_path) as f:
            self.dataset = json.load(f)

        self.version = self.dataset['version']
        self.raw_questions = self.dataset['questions']
        self.all_questions = [Question(**q) for q in self.raw_questions]
        self.mapped_questions = [q for q in self.all_questions if q.page is not None]

        self.guess_questions = [q for q in self.mapped_questions if q.fold == 'guess'+split]
        self.buzz_questions = [q for q in self.mapped_questions if q.fold == 'buzz'+split]




class QuizBowlDataset:
    def __init__(self, *, guesser = False, buzzer = False, split='train'):
        super().__init__()
        if not guesser and not buzzer:
            raise ValueError('Requesting a dataset which produces neither guesser or buzzer training data is invalid')

        if guesser and buzzer:
            print('Using QuizBowlDataset with guesser and buzzer training data, make sure you know what you are doing!')

        self.db = QantaDatabase(split)
        self.guesser = guesser
        self.buzzer = buzzer

    def data(self):
        questions = []
        if self.guesser:
            questions.extend(self.db.guess_questions)
        if self.buzzer:
            questions.extend(self.db.buzz_questions)

        return questions

#--- QUIZBOWL DATASET UTILITY FUNCTIONS END---

###You don't need to change anything in this class
class TfidfGuesser:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.i_to_ans = None

    def train(self, training_data) -> None:
        questions, answers = [], []
        for ques in training_data:
            questions.append(ques.sentences)
            answers.append(ques.page)

        answer_docs = defaultdict(str)
        for q, ans in zip(questions, answers):
            text = ' '.join(q)
            answer_docs[ans] += ' ' + text

        x_array = []
        y_array = []
        for ans, doc in answer_docs.items():
            x_array.append(doc)
            y_array.append(ans)

        """if self.i_to_ans != None:
            self.i_to_ans = self.i_to_ans.update({i: ans for i, ans in enumerate(y_array)})
        else:"""
        self.i_to_ans = {i: ans for i, ans in enumerate(y_array)}

        """if self.tfidf_vectorizer != None:
            self.tfidf_vectorizer.fit(x_array)
        else:"""
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_df=.9).fit(x_array)

        """if self.tfidf_matrix != None:
            self.tfidf_matrix ="""
        self.tfidf_matrix = self.tfidf_vectorizer.transform(x_array)

    def guess(self, questions: List[str], max_n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:
        representations = self.tfidf_vectorizer.transform(questions)
        guess_matrix = self.tfidf_matrix.dot(representations.T).T
        guess_indices = (-guess_matrix).toarray().argsort(axis=1)[:, 0:max_n_guesses]

        guesses = []
        for i in range(len(questions)):
            idxs = guess_indices[i]
            guesses.append([(self.i_to_ans[j], guess_matrix[i, j]) for j in idxs])

        return guesses

    def return_guess_prob(self, questions: List[str], max_n_guesses: Optional[int]):
        representations = self.tfidf_vectorizer.transform(questions)
        guess_matrix = self.tfidf_matrix.dot(representations.T).T
        guess_indices = (-guess_matrix).toarray().argsort(axis=1)[:, 0:max_n_guesses]

        guesses = []
        for i in range(len(questions)):
            idxs = guess_indices[i]
            guesses.append([(self.i_to_ans[j], guess_matrix[i, j]) for j in idxs])


    def save(self, guesser_model_path):
        with open(guesser_model_path, 'wb') as f:
            pickle.dump({
                'i_to_ans': self.i_to_ans,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'tfidf_matrix': self.tfidf_matrix
            }, f)

    @classmethod
    def load(cls, guesser_model_path):
        with open(guesser_model_path, 'rb') as f:
            params = pickle.load(f)
            guesser = TfidfGuesser()
            guesser.tfidf_vectorizer = params['tfidf_vectorizer']
            guesser.tfidf_matrix = params['tfidf_matrix']
            guesser.i_to_ans = params['i_to_ans']
            return guesser


###You don't need to change this funtion
def get_trained_guesser_model(tfidf_guesser, questions):
    print('Training the Guesser...')
    tfidf_guesser.train(questions)
    print('---Guesser is Trained and Ready to be Used---')
    return tfidf_guesser


###You don't need to change this funtion
def generate_ques_data_for_guesses(questions, char_skip = 50):
    ques_nums = []
    char_indices = []
    question_texts = []
    answers = []
    question_lens = []

    print("Preparing Data for Guessing; # of questions: " + str(len(questions)))

    for q in questions:
        qnums_temp, answers_temp, char_inds_temp, curr_ques_texts = [], [], [], []
        for text_run, char_ix in zip(*(q.runs(char_skip))):
            curr_ques_texts.append(text_run)
            qnums_temp.append(q.qanta_id)
            answers_temp.append(q.page)
            char_inds_temp.append(char_ix)
        ques_nums.append(qnums_temp)
        char_indices.append(char_inds_temp)
        question_texts.append(curr_ques_texts)
        answers.append(answers_temp)
        question_lens.append(len(curr_ques_texts))

    return ques_nums, answers, char_indices, question_texts, question_lens


###You don't need to change this funtion
def generate_guesses_and_scores(model, questions, max_guesses, char_skip = 50):
    #get the neccesary data
    qnums, answers, char_indices, ques_texts, ques_lens = generate_ques_data_for_guesses(questions, char_skip)
    print('Guessing...')

    guesses_and_scores = []
    for i in range(0, len(ques_texts), 250):
        try:
            q_texts_temp = ques_texts[i:i+250]
            q_lens = ques_lens[i:i+250]
        except:
            q_texts_temp = ques_texts[i:]
            q_lens = ques_lens[i:]

        #flatten
        q_texts_flattened = []
        for q in q_texts_temp:
            q_texts_flattened.extend(q)

        #store guesses for the flattened questions
        print('Guessing directly on %d text snippets together' %len(q_texts_flattened))
        flattened_guesses_scores = model.guess(q_texts_flattened, max_guesses)

        #de-flatten using question lengths, and add guesses and scores
        #(now corresponding to one question at a time) to the main list
        j = 0
        for k in q_lens:
            guesses_and_scores.append(flattened_guesses_scores[j:j+k])
            j = j + k

    assert len(guesses_and_scores)==len(ques_texts)

    print('Done Generating Guesses and Scores.')

    return qnums, answers, char_indices, ques_texts, ques_lens, guesses_and_scores


#You need to write code inside this function
def create_feature_vecs_and_labels(guesses_and_scores, answers, n_guesses):
    xs, ys = [], []

    for i in range(len(answers)):
        guesses_scores = guesses_and_scores[i]
        ans = answers[i]
        length = len(ans)
        labels = []
        prob_vec = []

        for i in range(length):
            labels.append(np.int(1) if ans[i] == guesses_scores[i][0][0] else np.int(0))

            temp = []
            for j in range(n_guesses):
                temp.append(guesses_scores[i][j][1])

            prob_vec.append(temp)

        xs.append(np.array(prob_vec))
        ys.append(np.array(np.int64(labels)))
    exs = list(zip(xs, ys))
    return exs


###You don't need to change this funtion
def batchify(batch):
    question_len = list()
    label_list = list()
    for ex in batch:
        question_len.append(len(ex[0]))
        label_list.append(ex[1])

    target_labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(y) for y in label_list], padding_value=-1).t()

    #dimension is dimension of every feature vector = n_guesses in this homework setting
    dim = batch[0][0].shape[1]

    #similar padding happens for the feature vectors, with vector of all zeros appended.
    x1 = torch.FloatTensor(len(question_len), max(question_len), dim).zero_()
    for i in range(len(question_len)):
        question_feature_vec = batch[i][0]
        vec = torch.FloatTensor(question_feature_vec)
        x1[i, :len(question_feature_vec)].copy_(vec)
    q_batch = {'feature_vec': x1, 'len': torch.FloatTensor(question_len), 'labels': target_labels}
    return q_batch


class QuestionDataset(Dataset):
    ###You don't need to change this funtion
    def __init__(self, examples):
        self.questions = []
        self.labels = []

        for qq, ll in examples:
            self.questions.append(qq)
            self.labels.append(ll)

    ###You don't need to change this funtion
    def __getitem__(self, index):
        return self.questions[index], self.labels[index]

    ###You don't need to change this funtion
    def __len__(self):
        return len(self.questions)

def ordering_sent(sent_list, key_word, guesser_model, n_guess):
    rankings = []
    for i in range(len(sent_list)):
        prob = guesser_model.guess([sent_list[i]], n_guess)
        prob = dict(prob[0])
        if key_word in prob:
            rankings.append((i, prob[key_word]))
        else:
            rankings.append((i, 0))

    ordered_sent = []
    rankings.sort(key = lambda tup: tup[1])
    for i, _ in rankings:
        ordered_sent.append(sent_list[i])

    return ordered_sent, rankings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options for RNN Buzzer')
    parser.add_argument('--guesser_model_path', type=str, default='tfidf.pickle', help='path for saving the trained guesser model')
    parser.add_argument('--guesser_train_limit', type=int, default=-1, help = 'Limit the data used to train the Guesser (total is around 90,000)')
    parser.add_argument('--guesser_saved_flag', type=bool, default=False, help='flag indicating use of saved guesser model or training one')

    args = parser.parse_args()

    if args.guesser_train_limit<0:
        train_guess_questions = QuizBowlDataset(guesser=True, split='train').data()
    else:
        train_guess_questions = QuizBowlDataset(guesser=True, split='train').data()[:args.guesser_train_limit]

    test_buzz_questions = QuizBowlDataset(buzzer=True, split='test').data()

    if args.guesser_saved_flag:
        guesser_model = TfidfGuesser().load(args.guesser_model_path)
    else:
        tfidf_guesser = TfidfGuesser()
        guesser_model = get_trained_guesser_model(tfidf_guesser, train_guess_questions)
        #guesser_model = get_trained_guesser_model(tfidf_guesser, dev_buzz_questions)
        #guesser_model = get_trained_guesser_model(tfidf_guesser, test_buzz_questions)
        guesser_model.save(args.guesser_model_path)
        print('Guesser Model Saved! Use --guesser_saved_flag=True when you next run the code to load the trained guesser directly.')


    questions, answers = [], []
    for ques in test_buzz_questions:
        questions.append(ques.sentences)
        answers.append(ques.page)

    for i in range(len(questions)):
        ordered_sent, rankings = ordering_sent(questions[i], answers[i], guesser_model, 1000)
        if sum([i[1] for i in rankings]) == 0:
            print("!!!!!")
        else:
            print([i[0] for i in rankings])
