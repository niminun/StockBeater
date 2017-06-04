import datetime
import tensorflow as tf

import nnutils
import MarketDataLoader


class StockBeater(object):

    def __init__(self, train_params):
        self.symbols = train_params.symbols
        self.params = train_params
        self.data_manager = MarketDataLoader.MarketDataLoader(self.symbols, train_params.option,
                                                              train_params.train_set_ratio,
                                                              train_params.records_per_sample,
                                                              train_params.fetch_data,
                                                              train_params.first_year)
        self.observed_batch = None
        self.day_after_batch = None
        self.decision = None
        self.train_step = None
        self.loss = None
        self.mean_profit = None
        self.is_train = tf.constant(False)
        self.keep_prob = 1.0
        self.sess = tf.InteractiveSession()
        self.init_nn_model()
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.params.summaries_dir + '/train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.params.summaries_dir + '/test')
        self.saver = tf.train.Saver()

    def learn(self):
        for t in range(self.params.num_iters):
            observed_batch, day_after_batch = self.data_manager.get_batch(self.params.train_batch_size, is_train=True)

            if t % 100 == 0:
                summary, loss = self.sess.run([self.merged, self.loss],
                                              feed_dict={self.observed_batch: observed_batch,
                                                         self.day_after_batch: day_after_batch,
                                                         self.keep_prob: 1.0})
                self.train_writer.add_summary(summary, t)
                print("Train: t = {0}, mean train loss = {1}".format(t, loss))

                if t % 1000 == 0:
                    self.saver.save(self.sess, "./snapshots/{0}_{1}".format(datetime.datetime.now().date(), t))
                    print("Test: t = {0}".format(t))
                    self.test(t)

            self.train_step.run(feed_dict={self.observed_batch: observed_batch,
                                           self.day_after_batch: day_after_batch,
                                           self.keep_prob: 0.6})

    def test(self, t):
        observed_batch, day_after_batch = self.data_manager.get_batch(self.params.test_batch_size, is_train=False)
        summary, loss, decision = self.sess.run([self.merged, self.loss, self.decision],
                                                feed_dict={self.observed_batch: observed_batch,
                                                           self.day_after_batch: day_after_batch,
                                                           self.keep_prob: 1.0})
        self.test_writer.add_summary(summary, t)
        print("mean test loss = {0}, decision:\n{1}".format(loss, decision))

    def predict(self, date_to_predict):
        observed = self.data_manager.get_data_for_prediction(self.params.records_per_sample, date_to_predict,
                                                             self.params.fetch_data_for_prediction)
        self.saver.restore(self.sess, self.params.use_model)
        prediction = self.decision.eval(feed_dict={self.observed_batch: observed, self.keep_prob: 1.0})
        return prediction

    def init_nn_model(self):

        # Convolutional Layers output channels sizes
        C1 = 512
        C2 = 512

        # fully connected Layers sizes
        FC1 = 512
        FC2 = 512

        # place holder that determines if it is a train iteration or a test iteration
        self.is_train = tf.placeholder(tf.bool)

        # calculations for sizes in the graph
        in_height = self.data_manager.data.shape[0]
        in_width = self.params.records_per_sample
        out_len = in_height

        # arranging the input
        self.observed_batch = tf.placeholder(dtype=tf.float32, shape=[None, in_height, in_width])
        x = tf.reshape(self.observed_batch, [-1, in_height, in_width, 1])

        # init variables
        k_conv1_shape = (in_height, 5)
        k_conv2_shape = (1, 3)
        sz_after_convs = (in_height - k_conv1_shape[0] + 1 - k_conv2_shape[0] + 1) * \
                         (in_width - k_conv1_shape[1] + 1 - k_conv2_shape[1] + 1) * C2

        # with tf.name_scope('conv1'):
        #     with tf.name_scope('weights'):
        #         W_conv1 = nnutils.weight_variable([k_conv1_shape[0], k_conv1_shape[1], 1, C1])
        #         nnutils.variable_summaries(W_conv1)
        #     with tf.name_scope('biases'):
        #         b_conv1 = nnutils.bias_variable([C1])
        #         nnutils.variable_summaries(b_conv1)
        #
        # with tf.name_scope('conv2'):
        #     with tf.name_scope('weights'):
        #         W_conv2 = nnutils.weight_variable([k_conv2_shape[0], k_conv2_shape[1], C1, C2])
        #         nnutils.variable_summaries(W_conv2)
        #     with tf.name_scope('biases'):
        #         b_conv2 = nnutils.bias_variable([C2])
        #         nnutils.variable_summaries(b_conv2)

        # with tf.name_scope('fc1'):
        #     with tf.name_scope('weights'):
        #         W_fc1 = nnutils.weight_variable([sz_after_convs, FC1])
        #         nnutils.variable_summaries(W_fc1)
        #     with tf.name_scope('biases'):
        #         b_fc1 = nnutils.bias_variable([FC1])
        #         nnutils.variable_summaries(b_fc1)
        #
        # with tf.name_scope('fc2'):
        #     with tf.name_scope('weights'):
        #         W_fc2 = nnutils.weight_variable([FC1, FC2])
        #         nnutils.variable_summaries(W_fc2)
        #     with tf.name_scope('biases'):
        #         b_fc2 = nnutils.bias_variable([FC2])
        #         nnutils.variable_summaries(b_fc2)
        #
        # with tf.name_scope('fcfinal'):
        #     with tf.name_scope('weights'):
        #         W_fc3 = nnutils.weight_variable([FC2, out_len])
        #         nnutils.variable_summaries(W_fc3)
        #     with tf.name_scope('biases'):
        #         b_fc3 = nnutils.bias_variable([out_len])
        #         nnutils.variable_summaries(b_fc3)


        # prediction:
        # convolutions
        conv1_out = nnutils.conv_bn_layer(x,
                                          [k_conv1_shape[0], k_conv1_shape[1], 1, C1],
                                          self.is_train, "conv_1")

        conv2_out = nnutils.conv_bn_layer(conv1_out,
                                          [k_conv2_shape[0], k_conv2_shape[1], C1, C2],
                                          self.is_train, "conv_2")

        # fully-connected
        conv2_flat = tf.reshape(conv2_out, [-1, sz_after_convs])
        fc1_out = nnutils.fullyconnected_bn_layer(conv2_flat,
                                                  [sz_after_convs, FC1],
                                                  self.is_train, "fully_connected_1")

        fc2_out = nnutils.fullyconnected_bn_layer(fc1_out,
                                                  [FC1, FC2],
                                                  self.is_train, "fully_connected_2")

        fc_final = nnutils.fullyconnected_bn_layer(fc2_out,
                                                   [FC2, out_len],
                                                   self.is_train, "fully_connected_final")

        # train step
        with tf.name_scope('decision'):
            self.decision = tf.nn.softmax(fc_final)
            tf.summary.histogram('histogram', self.decision)

        self.day_after_batch = tf.placeholder(dtype=tf.float32, shape=[None, in_height])
        last_observed = self.observed_batch[:, :, -1]
        change_rate = (self.day_after_batch - last_observed) / last_observed  # for matmul
        _y = tf.argmax(change_rate, axis=-1)
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=_y, logits=fc_final))

        with tf.name_scope('profit'):
            decision_mat = tf.reshape(self.decision, [-1, 1, out_len])  # for matmul
            change_rate_mat = tf.reshape(change_rate, [-1, 1, out_len])  # for matmul
            self.mean_profit = tf.reduce_mean(tf.matmul(change_rate_mat, decision_mat, transpose_b=True))
            tf.summary.scalar('mean_profit', self.mean_profit)

        train_vars = tf.trainable_variables()

        with tf.name_scope('loss'):
            self.loss = cross_entropy + tf.add_n([self.params.lmbda * tf.nn.l2_loss(v) for v in train_vars])
            tf.summary.scalar('loss', self.loss)

        self.train_step = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate).minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())
