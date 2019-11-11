# coding: utf-8
import os
import json
import pickle
import nltk
import numpy as np
from nltk.tokenize import WordPunctTokenizer
from collections import OrderedDict

import csv
from collections import defaultdict
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = WordPunctTokenizer()
vocab = {}

def build_vocab(vocab_path, yelp_json_path):
    global vocab
    if os.path.exists(vocab_path):
        vocab_file = open(vocab_path, 'r')
        vocab = json.load(vocab_file, object_pairs_hook=OrderedDict)
        print("load vocab finish!")
    else:
        word_freq = defaultdict(int)
        with open(yelp_json_path, 'r') as f:
            for line in f:
                review = json.loads(line)
                words = word_tokenizer.tokenize(review['text'])
                for word in words:
                    word_freq[word] += 1
            print("load finished")

        i = 1
        vocab['UNKNOW_TOKEN'] = 0
        for word, freq in word_freq.items():
            if freq > 3:  # 频率为3次之上
                vocab[word] = i
                i += 1
        vocab = sorted(vocab.items(), key=lambda e: e[1])
        json.dump((vocab), open(vocab_path, 'w'))

    return vocab


def load_dataset(yelp_json_path, max_sent_in_doc, max_word_in_sent):
    global vocab
    yelp_data_path = './data/out4.2_shuffle0_data.pickle'
    vocab_path = './data/out4.2_shuffle0_vocab.json'
    doc_num = 11368
    if not os.path.exists(yelp_data_path):
        vocab = build_vocab(vocab_path, yelp_json_path)
        UNKNOWN = 0
        data_x = np.zeros([doc_num, max_sent_in_doc, max_word_in_sent])
        with open(yelp_json_path, 'r') as f:
            for line_index, line in enumerate(f):
                review = json.loads(line)
                sents = sent_tokenizer.tokenize(review['text'])
                doc = np.zeros([max_sent_in_doc, max_word_in_sent])
                for i, sent in enumerate(sents):
                    if i < max_sent_in_doc:
                        word_to_index = np.zeros([max_word_in_sent], dtype=int)
                        for j, word in enumerate(word_tokenizer.tokenize(sent)):
                            if j < max_word_in_sent:
                                    word_to_index[j] = vocab.get(word, UNKNOWN)
                        doc[i] = word_to_index
                data_x[line_index] = doc
            pickle.dump((data_x), open(yelp_data_path, 'wb'))

    else:
        vocab = build_vocab(vocab_path, yelp_json_path)
        data_file = open(yelp_data_path, 'rb')
        data_x = pickle.load(data_file)


    length = len(data_x)
    file = open('data/gnd_shuffle%s.csv' % str(0), 'r')
    reader = csv.reader(file)
    lines = [row for row in reader]
    file_8 = open('data/gnd_shuffle%s_8.csv' % str(0), 'r')
    reader_8 = csv.reader(file_8)
    lines_8 = [row for row in reader_8]
    train_x, dev_x = data_x[:int(length * 0.9)-7], data_x[int(length * 0.9):-1]
    train_y, dev_y = lines[:int(length * 0.9)-7], lines[int(length * 0.9):-1]
    train_y_8, dev_y_8 = lines_8[:int(length * 0.9)-7], lines_8[int(length * 0.9):-1]
   
    return train_x, dev_x, length, vocab, train_y, dev_y, train_y_8, dev_y_8

if __name__ == '__main__':
    load_dataset("./data/out4.2_shuffle0.json", 5, 25)

