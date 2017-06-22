import tensorflow as tf


def weight_variable(_shape, _std=0.001):
    initial = tf.truncated_normal(shape=_shape, stddev=_std)
    return tf.Variable(initial)


def bias_variable(_shape, _std=0.001):
    initial = tf.constant(shape=_shape, stddev=_std)
    return tf.Variable(initial)


def conv2d(_x, _W):
    return tf.nn.conv2d(_x, _W, strides=[1, 1, 1, 1], padding='VALID')


def fullyconnected(_x, _W, _b):
    return tf.matmul(_x, _W) + _b


def conv_layer(input_tensor, weights_shape, layer_name):
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable(weights_shape)
            variable_summaries(weights)
        with tf.name_scope('conv'):
            pre_activation = conv2d(input_tensor, weights)
            tf.summary.histogram('pre_activation', pre_activation)
        with tf.name_scope('activation'):
            activations = tf.nn.relu(pre_activation)
            tf.summary.histogram('activations', activations)
        return activations


def fullyconnected_layer(input_tensor, weights_shape, layer_name):
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable(weights_shape)
            variable_summaries(weights)
        with tf.name_scope('fully_connected'):
            pre_activation = tf.matmul(input_tensor, weights)
            tf.summary.histogram('pre_activation', pre_activation)
        with tf.name_scope('activation'):
            activations = tf.nn.relu(pre_activation)
            tf.summary.histogram('activations', activations)
        return activations


def conv_bn_layer(input_tensor, weights_shape, is_train, layer_name):
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable(weights_shape)
            variable_summaries(weights)
        with tf.name_scope('conv'):
            pre_norm = conv2d(input_tensor, weights)
            tf.summary.histogram('pre_normalization', pre_norm)
        with tf.name_scope('batch_norm'):
            pre_activation = tf.layers.batch_normalization(pre_norm, training=is_train, momentum=0.999)
            tf.summary.histogram('pre_activation', pre_activation)
        with tf.name_scope('activation'):
            activations = tf.nn.relu(pre_activation)
            tf.summary.histogram('activations', activations)
        return activations


def fullyconnected_bn_layer(input_tensor, weights_shape, is_train, layer_name):
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable(weights_shape)
            variable_summaries(weights)
        with tf.name_scope('fully_connected'):
            pre_norm = tf.matmul(input_tensor, weights)
            tf.summary.histogram('pre_normalization', pre_norm)
        with tf.name_scope('batch_norm'):
            pre_activation = tf.layers.batch_normalization(pre_norm, training=is_train, momentum=0.999)
            tf.summary.histogram('pre_activation', pre_activation)
        with tf.name_scope('activation'):
            activations = tf.nn.relu(pre_activation)
            tf.summary.histogram('activations', activations)
        return activations


def variable_summaries(train_var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(train_var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(train_var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(train_var))
        tf.summary.scalar('min', tf.reduce_min(train_var))
        tf.summary.histogram('histogram', train_var)