import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
import networkx as nx


class STNE(object):
    ############
    def encoder(self,X,layers):
        for i in range(layers - 1):
            name_W = 'encoder_W_' + str(i)
            name_b = 'encoder_b_' + str(i)
            X = tf.nn.tanh(tf.matmul(X, self.W[name_W]) + self.b[name_b])
        return X

    def decoder(self,X,layers):
        for i in range(layers - 1):
            name_W = 'decoder_W_' + str(i)
            name_b = 'decoder_b_' + str(i)
            X = tf.nn.tanh(tf.matmul(X, self.W[name_W]) + self.b[name_b])
        return X

    def make_autoencoder_loss(self,X_new,X_re):
        def get_autoencoder_loss(X, newX):
            return tf.reduce_sum(tf.pow((newX - X), 2))

        def get_reg_loss(weights, biases):
            reg = tf.add_n([tf.nn.l2_loss(w) for w in weights.values()])
            reg += tf.add_n([tf.nn.l2_loss(b) for b in biases.values()])
            return reg
        loss_autoencoder = get_autoencoder_loss(X_new, X_re)
        loss_reg = get_reg_loss(self.W, self.b)
        return self.config.alpha * loss_autoencoder + self.config.reg * loss_reg
    ################
    def construct_traget_neighbors(self, nx_G, X, mode='EMN'):
        # construct target neighbor feature matrix
        X_target = np.zeros(X.shape)
        nodes = nx_G.nodes()
        if mode == 'OWN':
            # autoencoder for reconstructing itself
            return X
        elif mode == 'EMN':
            # autoencoder for reconstructing Elementwise Median Neighbor
            for node in nodes:
                neighbors = list(nx_G.neighbors(node))
                if len(neighbors) == 0:
                    X_target[node] = X[node]
                else:
                    temp = np.array(X[node])
                    for n in neighbors:
                        # # if FLAGS.weighted:
                        # #     # weighted sum
                        # #     # temp = np.vstack((temp, X[n] * edgeWeight))
                        # #     pass
                        # else:
                        temp = np.vstack((temp, X[n]))
                    temp = np.median(temp, axis=0)
                    X_target[node] = temp
            return X_target
    #################
    def __init__(self,config, hidden_dim, nx_G, X_1,node_num, fea_dim, seq_len, attention_size,
                 depth=1, node_fea=None, node_fea_trainable=False):
        self.node_num, self.fea_dim, self.seq_len = node_num, fea_dim, seq_len
        self.attention_size = attention_size
        self.nx_G = nx_G
        self.X_1 = X_1
        self.config = config
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.input_seqs = tf.placeholder(tf.int32, shape=(None, self.seq_len), name='input_seq')
        print(self.input_seqs,'AZAZAZAAAAAAAAAAAAAAA')
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        X_target = self.construct_traget_neighbors(self.nx_G, self.X_1, mode='EMN')
        print(X_target, '222222222222222222222222')
        X_target = tf.constant(X_target, dtype=tf.float32)
        self.layers = len(config.struct)
        struct = config.struct
        self.W = {}
        self.b = {}
        # encode module
        for i in range(self.layers - 1):
            name_W = 'encoder_W_' + str(i)
            name_b = 'encoder_b_' + str(i)
            print(struct[i], '55555555555555555555')
            print(struct[i + 1], '6666666666666666666')
            self.W[name_W] = tf.get_variable(name_W, [struct[i], struct[i + 1]],
                                             initializer=tf.contrib.layers.xavier_initializer())
            print(self.W[name_W], 'AAAAAAAAAAAAAAAAAAAAAAA')
            self.b[name_b] = tf.get_variable(name_b, [struct[i + 1]], initializer=tf.zeros_initializer())
            print(self.b[name_b], 'SSSSSSSSSSSSSSSSSSSSS')
        # decode module
        struct.reverse()
        for i in range(self.layers - 1):
            name_W = 'decoder_W_' + str(i)
            name_b = 'decoder_b_' + str(i)
            self.W[name_W] = tf.get_variable(name_W, [struct[i], struct[i + 1]],
                                             initializer=tf.contrib.layers.xavier_initializer())
            self.b[name_b] = tf.get_variable(name_b, [struct[i + 1]], initializer=tf.zeros_initializer())
        config.struct.reverse()
        ############## define input ###################
        self.Y1 = self.encoder(self.X_1,self.layers)
        self.X1_reconstruct = self.decoder(self.Y1, self.layers)
        self.loss_autoencoder_1 = self.make_autoencoder_loss(X_target, self.X1_reconstruct)
        input_seq_embed = tf.nn.embedding_lookup(self.Y1, self.input_seqs, name='input_embed_lookup')
        print(input_seq_embed, '44444444444444444NNNNNNNNNNNNNNNNNN4')

        # encoder
        encoder_cell_fw_0 = tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_dim), output_keep_prob=1 - self.dropout)
        encoder_cell_bw_0 = tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_dim), output_keep_prob=1 - self.dropout)
        if depth == 1:
            encoder_cell_fw_all = tf.contrib.rnn.MultiRNNCell([encoder_cell_fw_0])
            encoder_cell_bw_all = tf.contrib.rnn.MultiRNNCell([encoder_cell_bw_0])
        else:
            encoder_cell_fw_1 = tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_dim), output_keep_prob=1 - self.dropout)
            encoder_cell_bw_1 = tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_dim), output_keep_prob=1 - self.dropout)

            encoder_cell_fw_all = tf.contrib.rnn.MultiRNNCell([encoder_cell_fw_0] + [encoder_cell_fw_1] * (depth - 1))
            encoder_cell_bw_all = tf.contrib.rnn.MultiRNNCell([encoder_cell_bw_0] + [encoder_cell_bw_1] * (depth - 1))

        encoder_outputs, encoder_final = bi_rnn(encoder_cell_fw_all, encoder_cell_bw_all, inputs=input_seq_embed,
                                                dtype=tf.float32)
        ######################Attention
        self.encoder_output = tf.concat(encoder_outputs, 2)
        input_shape = self.encoder_output.shape  # (batch_size, sequence_length, hidden_size)
        sequence_size = input_shape[1].value
        hidden_size = input_shape[2].value
        attention_w = tf.Variable(tf.truncated_normal([hidden_size, self.attention_size], stddev=0.1),
                                  name='attention_w')
        attention_b = tf.Variable(tf.constant(0.1, shape=[self.attention_size]), name='attention_b')
        attention_u = tf.Variable(tf.truncated_normal([self.attention_size], stddev=0.1), name='attention_u')
        z_list = []
        for t in range(sequence_size):
            u_t = tf.tanh(tf.matmul(self.encoder_output[:, t, :], attention_w) + tf.reshape(attention_b, [1, -1]))
            z_t = tf.matmul(u_t, tf.reshape(attention_u, [-1, 1]))
            z_list.append(z_t)
        # Transform to batch_size * sequence_size

        attention_z = tf.concat(z_list, axis=1)
        alpha = tf.nn.softmax(attention_z)

        # print((self.encoder_output * tf.reshape(self.alpha, [-1, sequence_size, 1])), '++++++++++++++++++++++')
        attention_output = self.encoder_output * tf.reshape(alpha, [-1, sequence_size, 1])

        final_output = tf.nn.dropout(attention_output, self.keep_prob)
        ######################
        c_fw_list, h_fw_list, c_bw_list, h_bw_list = [], [], [], []
        for d in range(depth):
            (c_fw, h_fw) = encoder_final[0][d]
            (c_bw, h_bw) = encoder_final[1][d]
            c_fw_list.append(c_fw)
            h_fw_list.append(h_fw)
            c_bw_list.append(c_bw)
            h_bw_list.append(h_bw)

        decoder_init_state = tf.concat(c_fw_list + c_bw_list, axis=-1), tf.concat(h_fw_list + h_bw_list, axis=-1)
        decoder_cell = tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_dim * 2), output_keep_prob=1 - self.dropout)
        decoder_init_state = LSTMStateTuple(
            tf.layers.dense(decoder_init_state[0], units=hidden_dim * 2, activation=None),
            tf.layers.dense(decoder_init_state[1], units=hidden_dim * 2, activation=None))
        encoder_output_T = tf.transpose(final_output, [1, 0, 2])
        new_state = decoder_init_state
        outputs_list = []
        for i in range(seq_len):
            new_output, new_state = decoder_cell(tf.zeros(shape=tf.shape(encoder_output_T)[1:]), new_state)  # None
            outputs_list.append(new_output)

        decoder_outputs = tf.stack(outputs_list, axis=0)  # seq_len * batch_size * hidden_dim
        decoder_outputs = tf.transpose(decoder_outputs, [1, 0, 2])  # batch_size * seq_len * hidden_dim
        self.decoder_outputs = decoder_outputs
        output_preds = tf.layers.dense(decoder_outputs, units=self.node_num, activation=None)
        loss_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_seqs, logits=output_preds)
        self.loss_ce = tf.reduce_mean(loss_ce, name='loss_ce')
        self.toll_loss=config.alpha*self.loss_ce+config.reg*self.loss_autoencoder_1
        self.train_op = tf.train.RMSPropOptimizer(config.sg_learning_rate).minimize(self.toll_loss)


class STNEConv(object):
    def conv_pool(self, in_tensor, filter_size, num_filters, s_length, embedding_size=256):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            # Convolution Layer
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            # None, seq_len, word_dim, 1
            conv = tf.nn.conv2d(in_tensor, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, s_length - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            # print(pooled)
        return pooled  # None, 1, 1, num_filters

    def __init__(self, hidden_dim, node_num, fea_dim, seq_len, contnt_len, num_filters, word_dim,
                 vocab_size, attention_size, depth=1, filter_sizes=[2, 4, 8]):
        self.node_num, self.fea_dim = node_num, fea_dim
        self.attention_size = attention_size
        self.seq_len = seq_len
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.input_seqs = tf.placeholder(tf.int32, shape=(None, self.seq_len), name='input_seq')
        self.input_seq_content = tf.placeholder(tf.int32, shape=(None, self.seq_len, contnt_len),
                                                name='input_seq_content')
        self.dropout_rnn = tf.placeholder(tf.float32, name='dropout_rnn')
        self.dropout_word = tf.placeholder(tf.float32, name='dropout_word')
        self.word_embeds_W = tf.Variable(initial_value=tf.random_uniform(shape=(vocab_size, word_dim)),
                                         name='content_embed', trainable=True)

        contnt_embeds = tf.nn.embedding_lookup(self.word_embeds_W, self.input_seq_content, name='input_content_embed')
        contnt_embeds = tf.reshape(contnt_embeds, [-1, contnt_len, word_dim, 1])
        print(contnt_embeds, '1111111111111111111111')
        pooled = []
        for fsize in filter_sizes:
            # batch*seq_len, 1, num_filters
            tmp = self.conv_pool(contnt_embeds, fsize, num_filters, contnt_len, word_dim)
            pooled.append(tf.reshape(tmp, [-1, self.seq_len, num_filters]))
        input_seq_embed = tf.concat(pooled, axis=-1)  # batch, seq_len, num_filters*len(filter_sizes)
        input_seq_embed = tf.nn.dropout(input_seq_embed, keep_prob=1 - self.dropout_word)
        print(input_seq_embed, '22222222222222222222222')
        encoder_cell_fw_0 = tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_dim), output_keep_prob=1 - self.dropout_rnn)
        encoder_cell_bw_0 = tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_dim), output_keep_prob=1 - self.dropout_rnn)
        if depth == 1:
            encoder_cell_fw_all = tf.contrib.rnn.MultiRNNCell([encoder_cell_fw_0])
            encoder_cell_bw_all = tf.contrib.rnn.MultiRNNCell([encoder_cell_bw_0])
        else:
            encoder_cell_fw_1 = tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_dim),
                                                              output_keep_prob=1 - self.dropout_rnn)
            encoder_cell_bw_1 = tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_dim),
                                                              output_keep_prob=1 - self.dropout_rnn)

            encoder_cell_fw_all = tf.contrib.rnn.MultiRNNCell([encoder_cell_fw_0] + [encoder_cell_fw_1] * (depth - 1))
            encoder_cell_bw_all = tf.contrib.rnn.MultiRNNCell([encoder_cell_bw_0] + [encoder_cell_bw_1] * (depth - 1))

        encoder_outputs, encoder_final = bi_rnn(encoder_cell_fw_all, encoder_cell_bw_all, inputs=input_seq_embed,
                                                dtype=tf.float32)
        ############################
        self.encoder_output = tf.concat(encoder_outputs, 2)
        input_shape = self.encoder_output.shape  # (batch_size, sequence_length, hidden_size)
        sequence_size = input_shape[1].value
        hidden_size = input_shape[2].value
        attention_w = tf.Variable(tf.truncated_normal([hidden_size, self.attention_size], stddev=0.1),
                                  name='attention_w')
        attention_b = tf.Variable(tf.constant(0.1, shape=[self.attention_size]), name='attention_b')
        attention_u = tf.Variable(tf.truncated_normal([self.attention_size], stddev=0.1), name='attention_u')
        z_list = []
        for t in range(sequence_size):
            u_t = tf.tanh(tf.matmul(self.encoder_output[:, t, :], attention_w) + tf.reshape(attention_b, [1, -1]))
            z_t = tf.matmul(u_t, tf.reshape(attention_u, [-1, 1]))
            z_list.append(z_t)
        # Transform to batch_size * sequence_size

        attention_z = tf.concat(z_list, axis=1)
        self.alpha = tf.nn.softmax(attention_z)

        # print((self.encoder_output * tf.reshape(self.alpha, [-1, sequence_size, 1])), '++++++++++++++++++++++')
        attention_output = self.encoder_output * tf.reshape(self.alpha, [-1, sequence_size, 1])

        self.final_output = tf.nn.dropout(attention_output, self.keep_prob)
        ####################################
        c_fw_list, h_fw_list, c_bw_list, h_bw_list = [], [], [], []

        for d in range(depth):
            (c_fw, h_fw) = encoder_final[0][d]
            (c_bw, h_bw) = encoder_final[1][d]
            c_fw_list.append(c_fw)
            h_fw_list.append(h_fw)
            c_bw_list.append(c_bw)
            h_bw_list.append(h_bw)

        decoder_init_state = tf.concat(c_fw_list + c_bw_list, axis=-1), tf.concat(h_fw_list + h_bw_list, axis=-1)
        decoder_cell = tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_dim * 2), output_keep_prob=1 - self.dropout_rnn)
        decoder_init_state = LSTMStateTuple(
            tf.layers.dense(decoder_init_state[0], units=hidden_dim * 2, activation=None),
            tf.layers.dense(decoder_init_state[1], units=hidden_dim * 2, activation=None))

        # self.encoder_output = tf.concat(encoder_outputs, axis=-1)
        # encoder_output_T = tf.transpose(self.encoder_output, [1, 0, 2])  # h
        encoder_output_T = tf.transpose(self.final_output, [1, 0, 2])
        new_state = decoder_init_state
        outputs_list = []
        for i in range(seq_len):
            new_output, new_state = decoder_cell(tf.zeros(shape=tf.shape(encoder_output_T)[1:]), new_state)  # None
            outputs_list.append(new_output)

        decoder_outputs = tf.stack(outputs_list, axis=0)  # seq_len * batch_size * hidden_dim
        decoder_outputs = tf.transpose(decoder_outputs, [1, 0, 2])  # batch_size * seq_len * hidden_dim
        ###########################
        input_shape = decoder_outputs.shape  # (batch_size, sequence_length, hidden_size)
        sequence_size = input_shape[1].value
        hidden_size = input_shape[2].value
        attention_w = tf.Variable(tf.truncated_normal([hidden_size, self.attention_size], stddev=0.1),
                                  name='attention_w')
        attention_b = tf.Variable(tf.constant(0.1, shape=[self.attention_size]), name='attention_b')
        attention_u = tf.Variable(tf.truncated_normal([self.attention_size], stddev=0.1), name='attention_u')
        z_list = []
        for t in range(sequence_size):
            u_t = tf.tanh(tf.matmul(decoder_outputs[:, t, :], attention_w) + tf.reshape(attention_b, [1, -1]))
            z_t = tf.matmul(u_t, tf.reshape(attention_u, [-1, 1]))
            z_list.append(z_t)
        # Transform to batch_size * sequence_size

        attention_z = tf.concat(z_list, axis=1)
        self.alpha = tf.nn.softmax(attention_z)

        # print((self.encoder_output * tf.reshape(self.alpha, [-1, sequence_size, 1])), '++++++++++++++++++++++')
        attention_output = decoder_outputs * tf.reshape(self.alpha, [-1, sequence_size, 1])

        final_output_1 = tf.nn.dropout(attention_output, self.keep_prob)
        #####################
        self.decoder_output = final_output_1
        # decoder_outputs, _ = dynamic_rnn(decoder_cell, inputs=self.encoder_output, initial_state=decoder_init_state)
        output_preds = tf.layers.dense(final_output_1, units=self.node_num, activation=None)
        print(self.input_seqs, '@@@@@@@@@@@@@@@@@@@@')
        print(output_preds, '$$$$$$$$$$$$$$$$$$$$')
        loss_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_seqs, logits=output_preds)
        self.loss_ce = tf.reduce_mean(loss_ce, name='loss_ce')

        self.global_step = tf.Variable(1, name="global_step", trainable=False)
