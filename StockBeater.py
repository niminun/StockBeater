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

        # Affine Layers sizes
        A1 = 512
        A2 = 512

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

        with tf.name_scope('conv1'):
            with tf.name_scope('weights'):
                W_conv1 = nnutils.weight_variable([k_conv1_shape[0], k_conv1_shape[1], 1, C1])
                nnutils.variable_summaries(W_conv1)
            with tf.name_scope('biases'):
                b_conv1 = nnutils.bias_variable([C1])
                nnutils.variable_summaries(b_conv1)

        with tf.name_scope('conv2'):
            with tf.name_scope('weights'):
                W_conv2 = nnutils.weight_variable([k_conv2_shape[0], k_conv2_shape[1], C1, C2])
                nnutils.variable_summaries(W_conv2)
            with tf.name_scope('biases'):
                b_conv2 = nnutils.bias_variable([C2])
                nnutils.variable_summaries(b_conv2)

        sz_after_convs = (in_height - k_conv1_shape[0] + 1 - k_conv2_shape[0] + 1) * \
                         (in_width - k_conv1_shape[1] + 1 - k_conv2_shape[1] + 1) * C2

        with tf.name_scope('fc1'):
            with tf.name_scope('weights'):
                W_fc1 = nnutils.weight_variable([sz_after_convs, A1])
                nnutils.variable_summaries(W_fc1)
            with tf.name_scope('biases'):
                b_fc1 = nnutils.bias_variable([A1])
                nnutils.variable_summaries(b_fc1)

        with tf.name_scope('fc2'):
            with tf.name_scope('weights'):
                W_fc2 = nnutils.weight_variable([A1, A2])
                nnutils.variable_summaries(W_fc2)
            with tf.name_scope('biases'):
                b_fc2 = nnutils.bias_variable([A2])
                nnutils.variable_summaries(b_fc2)

        with tf.name_scope('fcfinal'):
            with tf.name_scope('weights'):
                W_fc3 = nnutils.weight_variable([A2, out_len])
                nnutils.variable_summaries(W_fc3)
            with tf.name_scope('biases'):
                b_fc3 = nnutils.bias_variable([out_len])
                nnutils.variable_summaries(b_fc3)

        # prediction:
        # convolutions
        h_conv1 = tf.nn.relu(nnutils.conv2d(x, W_conv1) + b_conv1)
        h_conv2 = tf.nn.relu(nnutils.conv2d(h_conv1, W_conv2) + b_conv2)

        # fully-connected
        self.keep_prob = tf.placeholder(tf.float32)
        h_conv2_flat = tf.reshape(h_conv2, [-1, sz_after_convs])
        h_conv2_drop = tf.nn.dropout(h_conv2_flat, self.keep_prob)
        h_fc1 = tf.nn.relu(tf.matmul(h_conv2_drop, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        with tf.name_scope('h_final'):
            h_final = tf.matmul(h_fc2, W_fc3) + b_fc3
            tf.summary.histogram('histogram', h_final)

        # train step
        with tf.name_scope('decision'):
            self.decision = tf.nn.softmax(h_final)
            tf.summary.histogram('histogram', self.decision)

        self.day_after_batch = tf.placeholder(dtype=tf.float32, shape=[None, in_height])
        last_observed = self.observed_batch[:, :, -1]
        change_rate = (self.day_after_batch - last_observed) / last_observed  # for matmul
        _y = tf.argmax(change_rate, axis=-1)
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=_y, logits=h_final))

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
