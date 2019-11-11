# coding: utf-8
import tensorflow as tf
import time
import os
import numpy as np
from data_helper import load_dataset
from HAN_model import HAN
import csv
from compute_SPK import compute_SPK
from avgprec import avgprec
from one_error import one_error
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 

def train_step(x_batch, y_batch, y_batch_8):
    feed_dict = {
        han.input_x: x_batch,
        han.input_y: y_batch,
        han.input_y_8: y_batch_8,
        han.max_sentence_num: 5,
        han.max_sentence_length: 25,
        han.batch_size: len(x_batch),
        han.drop: True
    }
    _, cost, FY_, alpha_sen, alpha_doc, word_encoded, doc_encoded, label_embeded, gc1_out, gc2_out, out_final = sess.run([train_op, loss,
                                                                                                         han.out,
                                                            han.alpha_sen, han.alpha_doc, han.word_encoded, han.doc_encoded,
                                                                                                         han.label_embeded, han.gc1_out, han.gc2_out,
                                                                                                         han.out_final
                                                                                                         ], feed_dict)
    return cost, FY_, alpha_sen, alpha_doc


def dev_step(x_batch, y_batch, y_batch_8):
    feed_dict = {
        han.input_x: x_batch,
        han.input_y: y_batch,
        han.input_y_8: y_batch_8,
        han.max_sentence_num: 5,
        han.max_sentence_length: 25,
        han.batch_size: len(x_batch),
        han.drop: False
    }

    cost, FY_, alpha_sen, alpha_doc = sess.run([loss, han.out, han.alpha_sen,
                                                         han.alpha_doc], feed_dict)

    return cost, FY_, alpha_sen, alpha_doc

m = 0
tf.flags.DEFINE_integer("vocab_size", 3741, "vocabulary size")  # 4582 4238 3737 3753
tf.flags.DEFINE_string("yelp_json_path", 'data/out4.2_shuffle%s.json'%str(m), "data directory")
tf.flags.DEFINE_integer("num_classes", 32, "number of classes")
tf.flags.DEFINE_integer("num_classes_8", 8, "number of classes")
tf.flags.DEFINE_integer("embedding_size", 300, "Dimensionality of character embedding (default: 200)")  # 500 320
tf.flags.DEFINE_integer("hidden_size", 75, "Dimensionality of GRU hidden layer (default: 50)")  # 50-75-10
tf.flags.DEFINE_integer("train_batch_size", 60, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("dev_batch_size", 60, "Batch Size (default: 64)")  # 60
tf.flags.DEFINE_integer("num_epochs", 300, "Number of training epochs (default: 50)")   # 200个epoch
tf.flags.DEFINE_integer("checkpoint_every", 2, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 1, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("max_sent_in_doc", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("max_word_in_sent", 25, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("evaluate_every", 50, "evaluate every this many batches")  # 50
tf.flags.DEFINE_float("grad_clip", 5, "grad clip to prevent gradient explode")
FLAGS = tf.flags.FLAGS


max_acc_train = 0
max_acc_test = 0
max_Avg_prec_train = 0
max_Avg_prec_test = 0
max_one_err_train = 100
max_one_err_test = 100
count_TaxoRead = 0

t_set = [0.004]
p_set = [0.2]


for t in t_set:
    for p in p_set:
        path_A_matrix = 'A_matrix_%s_%s.npy' % (str(t), str(p))
        A_matrix = np.load(path_A_matrix)
        train_x, dev_x, length, vocab, train_y, dev_y, train_y_8, dev_y_8 = load_dataset(FLAGS.yelp_json_path, FLAGS.max_sent_in_doc, FLAGS.max_word_in_sent)
        Y = train_y
        Y1 = dev_y
        Y_8 = train_y_8
        Y1_8 = dev_y_8
        N = FLAGS.train_batch_size
        print("data load finished")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        save_file = './data_A/checkpoint_dir/model_shuffle_%s_%s.ckpt'% (str(t), str(p))
        tf.reset_default_graph()
        count = 0
        count_train = 0
        count_dev = 0
        cost_train_epoch = 0
        acc_train_epoch = 0
        cost_dev_epoch = 0
        acc_dev_epoch = 0
        max_Avg_prec = 0
        max_gamma_Avg_prec = 0
        max_beta_Avg_prec = 0
        APK = []
        ASK = []
        APK1 = []
        ASK1 = []
        min_one_err = 1000000
        min_gamma_one_err = 0
        min_beta_one_err = 0
        max_S_2_train = 0
        max_S_2_test = 0
        with tf.Session(config=config) as sess:
            han = HAN(vocab_size=FLAGS.vocab_size, vocab_dict=vocab, A_matrix=A_matrix, num_classes=FLAGS.num_classes, num_classes_8=FLAGS.num_classes_8, sess=sess, embedding_size=FLAGS.embedding_size,
                      hidden_size=FLAGS.hidden_size)

            regularizer = tf.contrib.layers.l2_regularizer(0.1)
            reg_term = tf.contrib.layers.apply_regularization(regularizer)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=han.input_y, logits=han.out)) + reg_term

            predict = tf.argmax(han.out, axis=1, name='predict')
            label = tf.argmax(han.input_y, axis=1, name='label')
            acc = tf.reduce_mean(tf.cast(tf.equal(predict, label), tf.float32))

            global_step = tf.Variable(0, trainable=False)


            learning_rate_2 = 0.0006
            
            optimizer = tf.train.AdamOptimizer(learning_rate_2)
            train_op = optimizer.minimize(loss)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=FLAGS.num_checkpoints)

            for epoch in range(FLAGS.num_epochs):
                print('epoch: ', epoch + 1)
                for i in range(0, 10224, FLAGS.train_batch_size):  # 170*60 = 10200
                    x = train_x[i:i + FLAGS.train_batch_size]
                    y = train_y[i:i + FLAGS.train_batch_size]
                    y_8 = train_y_8[i:i + FLAGS.train_batch_size]
                    count_train = count_train + 1
                    cost_second, FY_, alpha_sen_, alpha_doc_ = train_step(x, y, y_8)
                    cost_train_epoch = cost_train_epoch + cost_second

                    
                    if i == 0:
                        FY = FY_
                        alpha_sen_train = alpha_sen_
                        alpha_doc_train = alpha_doc_
                    else:
                        FY = np.concatenate((FY, FY_), axis=0)
                        alpha_sen_train = np.concatenate((alpha_sen_train, alpha_sen_), axis=0)
                        alpha_doc_train = np.concatenate((alpha_doc_train, alpha_doc_), axis=0)

                for kk in [1, 3, 5]:
                    SK, PK = compute_SPK(Y, FY, kk)
                    AS = np.sum(SK) / np.size(FY, 0)
                    AP = np.mean(PK / kk)
                    APK.append(AP)
                    ASK.append(AS)
                train_total = np.size(Y, 0)
                fake_index = np.arange(train_total).reshape(train_total, 1)

                fake_gnd = np.hstack((fake_index, Y))
                fake_pred = np.hstack((fake_index, FY))

                Avg_prec = avgprec(fake_gnd, fake_pred)
                one_err = one_error(fake_gnd, fake_pred)
                cost_train_last = float(cost_train_epoch) / count_train
                print('train: Avg_prec: %f, one_err: %f, S@1,3,5: %f, %f, %f, p@1,3,5: %f, %f, %f'
                      % (Avg_prec, one_err, ASK[0], ASK[1], ASK[2], APK[0], APK[1], APK[2]))
                
                for i in range(0, 1140, FLAGS.dev_batch_size):  # 1168
                    x = dev_x[i:i + FLAGS.dev_batch_size]
                    y = dev_y[i:i + FLAGS.dev_batch_size]
                    y_8 = dev_y_8[i:i + FLAGS.dev_batch_size]
                    cost2, FY2_, alpha_sen_, alpha_doc_ = dev_step(x, y, y_8)
                    cost_dev_epoch = cost_dev_epoch + cost2
                    count_dev = count_dev + 1
                   
                    if i == 0:
                        FY2 = FY2_
                        alpha_sen_test = alpha_sen_
                        alpha_doc_test = alpha_doc_
                    else:
                        FY2 = np.concatenate((FY2, FY2_), axis=0)
                        alpha_sen_test = np.concatenate((alpha_sen_test, alpha_sen_), axis=0)
                        alpha_doc_test = np.concatenate((alpha_doc_test, alpha_doc_), axis=0)
                cost_dev_last = float(cost_dev_epoch) / count_dev  # ！！
                test_total = np.size(Y1, 0)
                fake_index1 = np.arange(test_total).reshape(test_total, 1)
                fake_gnd1 = np.hstack((fake_index1, Y1))
                fake_pred1 = np.hstack((fake_index1, FY2))
                Avg_prec1 = avgprec(fake_gnd1, fake_pred1)
                one_err1 = one_error(fake_gnd1, fake_pred1)
                for kk in [1, 3, 5]:
                    SK1, PK1 = compute_SPK(Y1, FY2, kk)
                    AS = np.sum(SK1) / np.size(FY2, 0)
                    AP = np.mean(PK1 / kk)
                    APK1.append(AP)
                    ASK1.append(AS)

                print('test: Avg_prec1: %f,  one_err1: %f, S1@1,3,5: %f, %f, %f, p1@1,3,5: %f, %f, %f'
                      % (Avg_prec1, one_err1, ASK1[0], ASK1[1], ASK1[2], APK1[0], APK1[1], APK1[2]))
                
                if Avg_prec > max_Avg_prec_train:
                    max_Avg_prec_train = Avg_prec
                    min_one_err_train = one_err
                    S_1_train = ASK[0]
                    S_3_train = ASK[1]
                    S_5_train = ASK[2]
                    p_1_train = APK[0]
                    p_3_train = APK[1]
                    p_5_train = APK[2]

                    with open("data/attention/alpha_sen_train.txt", 'w') as f11:
                        np.set_printoptions(threshold=10000000)
                        f11.write(str(alpha_sen_train))
                    with open("data/attention/alpha_doc_train.txt", 'w') as f12:
                        np.set_printoptions(threshold=10000000)
                        f12.write(str(alpha_doc_train))
                if Avg_prec1 >max_Avg_prec_test:
                    max_Avg_prec_test = Avg_prec1
                    min_one_err_test = one_err1
                    S_1_test = ASK1[0]
                    S_3_test = ASK1[1]
                    S_5_test = ASK1[2]
                    p_1_test = APK1[0]
                    p_3_test = APK1[1]
                    p_5_test = APK1[2]

                    with open("data/attention/alpha_sen_test.txt", 'w') as f21:
                        np.set_printoptions(threshold=10000000)
                        f21.write(str(alpha_sen_test))
                    with open("data/attention/alpha_doc_test.txt", 'w') as f22:
                        np.set_printoptions(threshold=10000000)
                        f22.write(str(alpha_doc_test))
                    with open("./data_A/checkpoint_dir/epoch.txt",'w') as f3:
                        f3.write(str(epoch + 1))
                    saver.save(sess, save_file)
                APK = []
                ASK = []
                APK1 = []
                ASK1 = []

            with open("data_A/final_result_out4.2_shuffle_t_%s_train.txt" % str(t), 'a') as ff1:
                ff1.write('t: %s\n' % (str(t)))
                ff1.write('p: %s\n' % (str(p)))
                ff1.write('max_Avg_prec_train: %s, min_one_err_train: %s\n' % (str(max_Avg_prec_train), str(min_one_err_train)))
                ff1.write('S@1,3,5: %s, %s, %s\n' % (str(S_1_train), str(S_3_train), str(S_5_train)))
                ff1.write('p@1,3,5: %s, %s, %s\n' % (str(p_1_train), str(p_3_train), str(p_5_train)))
            with open("data_A/final_result_out4.2_shuffle_t_%s_test.txt" % str(t), 'a') as ff2:
                ff2.write('t: %s\n' % (str(t)))
                ff2.write('p: %s\n' % (str(p)))
                ff2.write('max_Avg_prec_test: %s, min_one_err_test: %s\n' % (str(max_Avg_prec_test), str(min_one_err_test)))
                ff2.write('S@1,3,5: %s, %s, %s\n' % (str(S_1_test), str(S_3_test), str(S_5_test)))
                ff2.write('p@1,3,5: %s, %s, %s\n' % (str(p_1_test), str(p_3_test), str(p_5_test)))

            print('t: %s, p: %s' % (str(t), str(p)))
            print('max_Avg_prec_train: %f, min_one_err_train: %f, S@1,3,5: %f, %f, %f, p@1,3,5: %f, %f, %f'
                                                                    % (max_Avg_prec_train, min_one_err_train,
                                                                       S_1_train, S_3_train, S_5_train,
                                                                       p_1_train, p_3_train, p_5_train))

            print('max_Avg_prec_test: %f, min_one_err_test: %f, S@1,3,5: %f, %f, %f, p@1,3,5: %f, %f, %f'
                                                                    % (max_Avg_prec_test, min_one_err_test,
                                                                       S_1_test, S_3_test, S_5_test,
                                                                       p_1_test, p_3_test, p_5_test))
            max_Avg_prec_test = 0
            min_one_err_test = 10000
            S_1_test = 0
            S_3_test = 0
            S_5_test = 0
            p_1_test = 0
            p_3_test = 0
            p_5_test = 0
