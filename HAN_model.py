# coding:utf8
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
from collections import OrderedDict
np.set_printoptions(threshold=np.inf)
import json
import csv
path_label_vocab = 'train_vocab_label_origin.json'
f_vocab = open(path_label_vocab, 'r')
vocab_label1 = json.load(f_vocab)
vocab_label = sorted(vocab_label1.items(), key=lambda e: e[1])
f_vocab.close()

def length(sequences):
    used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
    seq_len = tf.reduce_sum(used, reduction_indices=1)
    return tf.cast(seq_len, tf.int32)

class HAN():

    def __init__(self, vocab_size, vocab_dict, A_matrix, num_classes, num_classes_8, sess, embedding_size=300, hidden_size=75):

        self.vocab_size = vocab_size                                    
        self.vocab_dict = vocab_dict                                    
        self.A_matrix = A_matrix
        self.num_classes = num_classes                                  
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.sess = sess
        self.num_classes_8 = num_classes_8
        with tf.name_scope('placeholder'):
            self.max_sentence_num = tf.placeholder(tf.int32, name='max_sentence_num')
            self.max_sentence_length = tf.placeholder(tf.int32, name='max_sentence_length')
            self.batch_size = tf.placeholder(tf.int32, name='batch_size')
            self.input_x = tf.placeholder(tf.int64, [None, None, None], name='input_x')  # ！！！
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')   # input_y 为本身属于哪个类别 float 64
            self.input_y_8 = tf.placeholder(tf.float32, [None, num_classes_8], name='input_y')   # input_y 为本身属于哪个类别 float 64
            self.drop = tf.placeholder(tf.bool, name='drop_control')


        self.number = 0
        self.vocab_label =  vocab_label
        self.in_channel = 512  # 512
        self.gc1_out_dim = 300  # 300
        self.gc2_out_dim = self.hidden_size*2

        word_embedded = self.load_bin_vec(vocab_dict)
        sent_vec, alpha_sen, word_encoded = self.sent2vec(word_embedded)
        doc_vec, alpha_doc, doc_encoded = self.doc2vec(sent_vec)
        self.doc_vec = doc_vec
   
        # GCN
        self.label_embeded = self.embedding_label(self.num_classes, self.in_channel, self.list_word, self.list_vec)
        self.gc1_out = self.GraphConvolution(self.label_embeded, self.in_channel, self.gc1_out_dim, self.A_matrix, name='gc1')
        self.gc2_out = self.GraphConvolution(self.gc1_out, self.gc1_out_dim, self.gc2_out_dim, self.A_matrix, name='gc2')
        self.out_final = self.dot_product(self.doc_vec, self.gc2_out)
        self.out = self.out_final

        self.word_embedded = word_embedded
        self.alpha_sen = alpha_sen
        self.alpha_doc = alpha_doc
        self.sen_vec = sent_vec
        self.doc_vec = doc_vec
        self.word_encoded = word_encoded
        self.doc_encoded = doc_encoded


    def embedding_label(self, label_num, in_channels, list_word, list_vec):
        vocab_2 = {}
        final_vocab_label = []
        final_vocab_label = tf.Variable(tf.random_uniform([label_num, in_channels], -0.1, 0.1), name='final_vocab_label')
        return final_vocab_label

    def GraphConvolution(self, input_2, in_dim, out_dim, A_matrix, name):
        with tf.variable_scope(name):
            weight = tf.Variable(tf.truncated_normal([in_dim, out_dim]), name='weight')
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, weight)
            support = tf.matmul(tf.cast(input_2, tf.float32), weight)
            output = tf.nn.leaky_relu(tf.matmul(tf.cast(A_matrix, tf.float32), support))

        return output

    def dot_product(self, representation, gc_output):
        with tf.name_scope("dot_product"):
            gc_output2 = tf.transpose(gc_output, [1, 0])
            out_final = tf.matmul(representation, gc_output2, name="predictions")
            params = tf.trainable_variables()
            for idx, v in enumerate(params):
                print("param {:3}:{:15}{}".format(idx, str(v.get_shape()), v.name))
        return out_final

    def load_bin_vec(self, vocab2):  
        final_vocab = tf.Variable(tf.random_uniform([len(vocab2), self.embedding_size], -0.1, 0.1), name='final_vocab')
        word_embedded = tf.nn.embedding_lookup(final_vocab, self.input_x)
        word_embedded = tf.cast(word_embedded, tf.float32)
        return word_embedded

    def sent2vec(self, word_embedded):
        with tf.name_scope("sent2vec"):
            word_embedded = tf.reshape(word_embedded, [-1, self.max_sentence_length, self.embedding_size])
            word_encoded = self.BidirectionalGRUEncoder(word_embedded, name='word_encoder')
            sent_vec, alpha_sen = self.AttentionLayer(word_encoded, name='word_attention')
        return sent_vec, alpha_sen, word_encoded

    def doc2vec(self, sent_vec):
        with tf.name_scope("doc2vec"):
            sent_vec = tf.reshape(sent_vec, [-1, self.max_sentence_num, self.hidden_size*2])  
            doc_encoded = self.BidirectionalGRUEncoder(sent_vec, name='sent_encoder')
            doc_vec, alpha_doc = self.AttentionLayer(doc_encoded, name='sent_attention')

        return doc_vec, alpha_doc, doc_encoded

    def BidirectionalGRUEncoder(self, inputs, name):
        with tf.variable_scope(name):
            if self.drop == 'True':
                GRU_cell_fw = rnn.GRUCell(self.hidden_size)
                GRU_cell_fw = tf.nn.rnn_cell.DropoutWrapper(GRU_cell_fw, 0.75)
                GRU_cell_bw = rnn.GRUCell(self.hidden_size)
                GRU_cell_bw = tf.nn.rnn_cell.DropoutWrapper(GRU_cell_bw, 0.75)
                ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw,
                                                                                     cell_bw=GRU_cell_bw,
                                                                                     inputs=inputs,
                                                                                     sequence_length=length(inputs),
                                                                                     dtype=tf.float32)
            else:
                GRU_cell_fw = rnn.GRUCell(self.hidden_size)
                GRU_cell_fw = tf.nn.rnn_cell.DropoutWrapper(GRU_cell_fw, 1)
                GRU_cell_bw = rnn.GRUCell(self.hidden_size)
                GRU_cell_bw = tf.nn.rnn_cell.DropoutWrapper(GRU_cell_bw, 1)
                ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw,
                                                                                     cell_bw=GRU_cell_bw,
                                                                                     inputs=inputs,
                                                                                     sequence_length=length(inputs),
                                                                                     dtype=tf.float32)
            outputs = tf.concat((fw_outputs, bw_outputs), 2)
            self.number = self.number + 1
        return outputs

    def AttentionLayer(self, inputs, name):
        with tf.variable_scope(name):
            u_context = tf.Variable(tf.truncated_normal([self.hidden_size * 2]), name='u_context')
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, u_context)
            self.sess.run(u_context.initializer)
            h = layers.fully_connected(inputs, self.hidden_size * 2, activation_fn=tf.nn.tanh)
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True), dim=1)
            atten_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
        return atten_output, alpha