"""
    @ file : network.py
    @ brief

    @ author : Younghyun Lee <yhlee109@gmail.com>
    @ date : 2018.01.25
    @ version : 1.0
"""
import tensorflow as tf


class BatchNorm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay =self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)


def conv_cond_concat(x, y):
    """
    Concatenate conditioning vector on feature map axis.
    :param x:
    :param y:
    :return:
    """
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def conv2d(input_, output_dim, name="conv2d"):
    """

    :param input_:
    :param output_dim:
    :param name:
    :return:
    """
    k_h = 5
    k_w = 5
    d_h = 2
    d_w = 2
    stddev = 0.02
    w_init = tf.truncated_normal_initializer(stddev=stddev)

    with tf.variable_scope(name):

        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], initializer=w_init)
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def deconv2d(input_, output_shape, name="deconv2d"):
    """

    :param input_:
    :param output_shape:
    :param name:
    :return:
    """

    k_h = 5
    k_w = 5
    d_h = 2
    d_w = 2
    stddev = 0.02
    w_init = tf.random_normal_initializer(stddev=stddev)

    with tf.variable_scope(name):

        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]], initializer=w_init)

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        return deconv


def lrelu(x, leak=0.2):
    """

    :param x:
    :param leak:
    :return:
    """
    return tf.maximum(x, leak*x)


def generator_mnist(z, y=None, reuse=False, is_training=True):
    """
    Generator G(z,y)
    :param z:
    :param y:
    :param reuse:
    :param is_training:
    :return:
    """
    with tf.variable_scope("Generator", reuse=reuse):

        # initializer
        #w_init = tf.contrib.layers.xavier_initializer()  # tf.contrib.layers.variance_scaling_initializer()
        w_init = tf.random_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(0.0)

        # init batch norm
        gen_bn0 = BatchNorm(name='gen_bn0')
        gen_bn1 = BatchNorm(name='gen_bn1')
        gen_bn2 = BatchNorm(name='gen_bn2')

        # parameters
        batch_size = 64
        y_dim = 10
        img_depth = 1

        # layer setting
        s_h, s_w = 28, 28
        s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
        s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

        gen_conv_dim = 64
        gen_fc_dim = 1024

        # input
        yb = tf.reshape(y, [batch_size, 1, 1, y_dim])
        z = tf.concat([z, y], 1)

        # 1st layer (linear)
        num_hidden1 = gen_fc_dim
        w0 = tf.get_variable('w0', [z.get_shape()[1], num_hidden1], initializer=w_init)
        b0 = tf.get_variable('b0', [num_hidden1], initializer=b_init)
        h0 = gen_bn0(tf.matmul(z, w0) + b0, train=is_training)
        h0 = tf.nn.relu(h0)
        h0 = tf.concat([h0, y], 1)

        # 2nd layer (linear)
        num_hidden2 = gen_conv_dim * 2 * s_h4 * s_w4
        w1 = tf.get_variable('w1', [h0.get_shape()[1], num_hidden2], initializer=w_init)
        b1 = tf.get_variable('b1', [num_hidden2], initializer=b_init)
        h1 = gen_bn1(tf.matmul(h0, w1) + b1, train=is_training)
        h1 = tf.nn.relu(h1)
        h1 = tf.reshape(h1, [batch_size, s_h4, s_w4, gen_conv_dim * 2])
        h1 = conv_cond_concat(h1, yb)

        # 3rd layer (conv)
        h2 = deconv2d(h1, [batch_size, s_h2, s_w2, gen_conv_dim * 2], name='g_h2')
        h2 = gen_bn2(h2, train=is_training)
        h2 = tf.nn.relu(h2)
        h2 = conv_cond_concat(h2, yb)

        return tf.nn.sigmoid(deconv2d(h2, [batch_size, s_h, s_w, img_depth], name='g_h3'))


def discriminator_mnist(image, y=None, reuse=False):
    """
    Discriminator D(x,y)
    :param image:
    :param y:
    :param reuse:
    :return:
    """
    with tf.variable_scope("Discriminator", reuse=reuse):

        # initializer
        # w_init = tf.contrib.layers.xavier_initializer()  # tf.contrib.layers.variance_scaling_initializer()
        w_init = tf.random_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(0.0)

        # batch_norm
        disc_bn1 = BatchNorm(name='disc_bn1')
        disc_bn2 = BatchNorm(name='disc_bn2')

        # parameters
        batch_size = 64
        y_dim = 10
        img_depth = 1

        disc_conv_dim = 64
        disc_fc_dim = 1024

        # input
        yb = tf.reshape(y, [batch_size, 1, 1, y_dim])
        x = conv_cond_concat(image, yb)

        # 1st layer (conv)
        h0 = conv2d(x, img_depth + y_dim, name='d_h0_conv')
        h0 = lrelu(h0)
        h0 = conv_cond_concat(h0, yb)

        # 2nd layer (conv)
        h1 = conv2d(h0, disc_conv_dim + y_dim, name='d_h1_conv')
        h1 = disc_bn1(h1)
        h1 = lrelu(h1)
        h1 = tf.reshape(h1, [batch_size, -1])
        h1 = tf.concat([h1, y], 1)

        # 3rd layer (linear)
        num_hidden2 = disc_fc_dim
        w2 = tf.get_variable('w2', [h1.get_shape()[1], num_hidden2], initializer=w_init)
        b2 = tf.get_variable('b2', [num_hidden2], initializer=b_init)
        h2 = disc_bn2(tf.matmul(h1, w2) + b2)
        h2 = lrelu(h2)
        h2 = tf.concat([h2, y], 1)

        # 4th layer (linear)
        num_hidden3 = 1
        w3 = tf.get_variable('w3', [h2.get_shape()[1], num_hidden3], initializer=w_init)
        b3 = tf.get_variable('b3', [num_hidden3], initializer=b_init)
        h3 = tf.matmul(h2, w3) + b3

        return tf.nn.sigmoid(h3), h3


