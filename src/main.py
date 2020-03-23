# encoding: utf-8
import time
import networkx as nx
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression

from models import STNE
from config import *
from classify import Classifier, read_node_label
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True


def read_node_features(filename):
    fea = []
    fin = open(filename, 'r')
    for l in fin.readlines():
        vec = l.split()
        fea.append(np.array([float(x) for x in vec[1:]]))
    fin.close()
    return np.array(fea, dtype='float32')


def read_node_sequences(filename):
    seq = []
    fin = open(filename, 'r')
    for l in fin.readlines():
        vec = l.split()
        seq.append(np.array([int(x) for x in vec]))
    fin.close()
    return np.array(seq)


def reduce_seq2seq_hidden_mean(seq, seq_h, node_num, seq_num, seq_len):
    node_dict = {}
    for i_seq in range(seq_num):
        for j_node in range(seq_len):
            nid = seq[i_seq, j_node]
            if nid in node_dict:
                node_dict[nid].append(seq_h[i_seq, j_node, :])
            else:
                node_dict[nid] = [seq_h[i_seq, j_node, :]]
    vectors = []
    for nid in range(node_num):
        vectors.append(np.average(np.array(node_dict[nid]), 0))
    return np.array(vectors)


def reduce_seq2seq_hidden_add(sum_dict, count_dict, seq, seq_h_batch, seq_len, batch_start):
    for i_seq in range(seq_h_batch.shape[0]):
        for j_node in range(seq_len):
            nid = seq[i_seq + batch_start, j_node]
            if nid in sum_dict:
                sum_dict[nid] = sum_dict[nid] + seq_h_batch[i_seq, j_node, :]
            else:
                sum_dict[nid] = seq_h_batch[i_seq, j_node, :]
            if nid in count_dict:
                count_dict[nid] = count_dict[nid] + 1
            else:
                count_dict[nid] = 1
    return sum_dict, count_dict


def reduce_seq2seq_hidden_avg(sum_dict, count_dict, node_num):
    vectors = []
    for nid in range(node_num):
        vectors.append(sum_dict[nid] / count_dict[nid])
    return np.array(vectors)


def node_classification(session, bs, seqne, sequences, seq_len, node_n, samp_idx, label, ratio):
    enc_sum_dict = {}
    node_cnt = {}
    s_idx, e_idx = 0, bs
    while e_idx < len(sequences):
        batch_enc = session.run(seqne.encoder_output,
                                feed_dict={seqne.input_seqs: sequences[s_idx: e_idx],
                                           seqne.dropout: 0, seqne.keep_prob: 0})
        enc_sum_dict, node_cnt = reduce_seq2seq_hidden_add(enc_sum_dict, node_cnt, sequences,
                                                       batch_enc.astype('float32'), seq_len, s_idx)

        s_idx, e_idx = e_idx, e_idx + bs

    if s_idx < len(sequences):
        batch_enc = session.run(seqne.encoder_output,
                                feed_dict={seqne.input_seqs: sequences[s_idx: len(sequences)],
                                            seqne.dropout: 0,
                                           seqne.keep_prob: 0})
        enc_sum_dict, node_cnt = reduce_seq2seq_hidden_add(enc_sum_dict, node_cnt, sequences,
                                                           batch_enc.astype('float32'), seq_len, s_idx)

    node_enc_mean = reduce_seq2seq_hidden_avg(sum_dict=enc_sum_dict, count_dict=node_cnt, node_num=node_n)
    lr = Classifier(vectors=node_enc_mean, clf=LogisticRegression())
    f1_micro, f1_macro = lr.split_train_evaluate(samp_idx, label, ratio)
    return f1_micro


def node_classification_d(session, bs, seqne, sequences, seq_len, node_n, samp_idx, label, ratio):
    enc_sum_dict = {}
    node_cnt = {}
    s_idx, e_idx = 0, bs
    while e_idx < len(sequences):
        batch_dec = session.run(seqne.decoder_outputs,
                                feed_dict={seqne.input_seqs: sequences[s_idx: e_idx], seqne.dropout: 0,
                                           seqne.keep_prob: 0})
        enc_sum_dict, node_cnt = reduce_seq2seq_hidden_add(enc_sum_dict, node_cnt, sequences,
                                                           batch_dec.astype('float32'), seq_len, s_idx)
        s_idx, e_idx = e_idx, e_idx + bs

    if s_idx < len(sequences):
        batch_dec = session.run(seqne.encoder_output,
                                feed_dict={seqne.input_seqs: sequences[s_idx: len(sequences)], seqne.dropout: 0,
                                           seqne.keep_prob: 0})
        enc_sum_dict, node_cnt = reduce_seq2seq_hidden_add(enc_sum_dict, node_cnt, sequences,
                                                           batch_dec.astype('float32'), seq_len, s_idx)

    node_enc_mean = reduce_seq2seq_hidden_avg(sum_dict=enc_sum_dict, count_dict=node_cnt, node_num=node_n)

    lr = Classifier(vectors=node_enc_mean, clf=LogisticRegression())
    f1_micro, f1_macro = lr.split_train_evaluate(samp_idx, label, ratio)
    return f1_micro


def node_classification_hd(session, bs, seqne, sequences, seq_len, node_n, samp_idx, label, ratio):
    enc_sum_dict = {}
    dec_sum_dict = {}
    node_cnt_enc = {}
    node_cnt_dec = {}
    s_idx, e_idx = 0, bs
    while e_idx < len(sequences):
        batch_enc = session.run(seqne.encoder_output,
                                feed_dict={seqne.input_seqs: sequences[s_idx: e_idx], seqne.dropout: 0,
                                           seqne.keep_prob: 0.})
        enc_sum_dict, node_cnt_enc = reduce_seq2seq_hidden_add(enc_sum_dict, node_cnt_enc, sequences,
                                                               batch_enc.astype('float32'), seq_len, s_idx)

        batch_dec = session.run(seqne.decoder_outputs,
                                feed_dict={seqne.input_seqs: sequences[s_idx: e_idx], seqne.dropout: 0,
                                           seqne.keep_prob: 0})
        dec_sum_dict, node_cnt_dec = reduce_seq2seq_hidden_add(dec_sum_dict, node_cnt_dec, sequences,
                                                               batch_dec.astype('float32'), seq_len, s_idx)
        s_idx, e_idx = e_idx, e_idx + bs

    if s_idx < len(sequences):
        batch_enc = session.run(seqne.encoder_output,
                                feed_dict={seqne.input_seqs: sequences[s_idx: e_idx], seqne.dropout: 0,
                                           seqne.keep_prob: 0})
        enc_sum_dict, node_cnt_enc = reduce_seq2seq_hidden_add(enc_sum_dict, node_cnt_enc, sequences,
                                                               batch_enc.astype('float32'), seq_len, s_idx)

        batch_dec = session.run(seqne.decoder_outputs,
                                feed_dict={seqne.input_seqs: sequences[s_idx: e_idx], seqne.dropout: 0,
                                           seqne.keep_prob: 0})
        dec_sum_dict, node_cnt_dec = reduce_seq2seq_hidden_add(dec_sum_dict, node_cnt_dec, sequences,
                                                               batch_dec.astype('float32'), seq_len, s_idx)

    node_enc_mean = reduce_seq2seq_hidden_avg(sum_dict=enc_sum_dict, count_dict=node_cnt_enc, node_num=node_n)
    node_dec_mean = reduce_seq2seq_hidden_avg(sum_dict=dec_sum_dict, count_dict=node_cnt_dec, node_num=node_n)

    node_mean = np.concatenate((node_enc_mean, node_dec_mean), axis=1)
    lr = Classifier(vectors=node_mean, clf=LogisticRegression())
    f1_micro, f1_macro = lr.split_train_evaluate(samp_idx, label, ratio)
    return f1_micro


def check_all_node_trained(trained_set, seq_list, total_node_num):
    for seq in seq_list:
        trained_set.update(seq)
    if len(trained_set) == total_node_num:
        return True
    else:
        return False


def read_graph(edgeFile):
    print('loading graph...')

    G = nx.read_edgelist(edgeFile, nodetype=int, create_using=nx.DiGraph())
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1

    # if not FLAGS.directed:
    #     G = G.to_undirected()

    return G


def graph_context_batch_iter(all_pairs, batch_size):
    while True:
        start_idx = np.random.randint(0, len(all_pairs) - batch_size)
        batch_idx = np.array(range(start_idx, start_idx + batch_size))
        batch_idx = np.random.permutation(batch_idx)
        batch = np.zeros(batch_size, dtype=np.int32)
        # labels = np.zeros((batch_size, 1), dtype=np.int32)
        batch[:] = all_pairs[batch_idx, 0]
        # labels[:, 0] = all_pairs[batch_idx, 1]
        yield batch


if __name__ == '__main__':
    folder = '../da/wiki/'
    fn = '../da/wiki/result.txt'
    dpt = 1  # Depth of both the encoder and the decoder layers (MultiCell RNN)
    h_dim = 500  # Hidden dimension of encoder LSTMs
    s_len = 10  # Length of input node sequence
    epc = 2  # Number of training epochs
    trainable = False  # Node features trainable or not
    dropout = 0.2  # Dropout ration
    clf_ratio = [0.1, 0.2, 0.3, 0.4, 0.5]  # Ration of training samples in subsequent classification
    # b_s = 128  # Size of batches
    lr = 0.001  # Learning rate of RMSProp
    keep_prob = 0.5
    attention_size = 1000
    # max_iters = 20000
    print_every_k_iterations = 100
    idx = 0

    loss_1 = 0
    loss_2 = 0
    config = Config()
    start = time.time()
    fobj = open(fn, 'w')
    X, Y = read_node_label(folder + 'labels.txt')
    node_fea = read_node_features(folder + 'wiki.features')
    node_seq = read_node_sequences(folder + 'node_sequences_10_10.txt')
    nx_G = read_graph(folder + 'wiki.edgelist')
    nodes = nx_G.nodes()
    N = len(nodes)
    X_1 = read_node_features(folder + 'wiki.features')
    print(node_fea.shape[1], 'CCCCCCCCCCCCCCCCCCCC')
    # with tf.Session() as sess:
    model = STNE(config, hidden_dim=h_dim, nx_G=nx_G, X_1=X_1, seq_len=s_len, attention_size=100, depth=dpt,
                 node_fea=node_fea, node_fea_trainable=trainable,
                 node_num=node_fea.shape[0], fea_dim=node_fea.shape[1])

    init = tf.global_variables_initializer()
    sess = tf.Session(config=config_tf)
    sess.run(init)
    trained_node_set = set()
    all_trained = False
    batch_size = config.b_s
    max_iters = config.max_iters
    # for epoch in range(epc):
    for epoch in range(epc):
        # start_idx, end_idx = 0, config.b_s
        print('Epoch,\tidx,\ttotal_loss,\t#Trained Nodes')
        for iter_cnt in range(max_iters):
            batch_index = next(graph_context_batch_iter(node_seq, batch_size))
            # while end_idx < len(node_seq):
            idx += 1
            _, loss_1_value = sess.run([model.train_op, model.toll_loss],
                                       feed_dict={model.input_seqs: node_seq[batch_index],
                                                  model.dropout: dropout, model.keep_prob: keep_prob})
            loss_1 += loss_1_value

            if not all_trained:
                all_trained = check_all_node_trained(trained_node_set, node_seq[batch_index],
                                                     node_fea.shape[0])
            # start_idx, end_idx = end_idx, end_idx + config.b_s
            if idx % 10 == 0:
                end = time.time()
                total_loss = loss_1 / idx
                print(epoch, '\t', idx, '\t', total_loss, '\t', len(trained_node_set))
                if all_trained:
                    f1_mi = []
                    for ratio in clf_ratio:
                        f1_mi.append(node_classification(session=sess, bs=config.b_s, seqne=model, sequences=node_seq,

                                                         seq_len=s_len, node_n=node_fea.shape[0], samp_idx=X,
                                                         label=Y, ratio=ratio))

                    print('idx ', idx)
                    fobj.write('idx ' + str(idx) + ' ')
                    for f1 in f1_mi:
                        print(f1)
                        fobj.write(str(f1) + ' ')
                    fobj.write('\n')

        minute = np.around((time.time() - start) / 60)
        print('\nepoch ', epoch, ' finished in ', str(minute), ' minutes\n')

        f1_mi = []
        for ratio in clf_ratio:
            f1_mi.append(
                node_classification(session=sess, bs=config.b_s, seqne=model, sequences=node_seq,
                                    seq_len=s_len,
                                    node_n=node_fea.shape[0], samp_idx=X, label=Y, ratio=ratio))

        fobj.write(str(epoch) + ' ')
        print('Classification results on current ')
        for f1 in f1_mi:
            print(f1)
            fobj.write(str(f1) + ' ')
        fobj.write('\n')
        minute = np.around((time.time() - start) / 60)

        fobj.write(str(minute) + ' minutes' + '\n')
        print('\nClassification finished in ', str(minute), ' minutes\n')

    fobj.close()
    minute = np.around((time.time() - start) / 60)
    print('Total time: ' + str(minute) + ' minutes')
