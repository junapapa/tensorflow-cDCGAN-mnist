"""
    @ file : dcgan.py
    @ brief

    @ author : Younghyun Lee <yhlee109@gmail.com>
    @ date : 2018.01.25
    @ version : 1.0
"""
from network import *
from trainer import *

class DCGAN(object):
    """

    """
    def __init__(self, sess, name, input_width=28, input_height=28, output_width=28, output_height=28,
                 y_dim=None, z_dim=100, img_depth=3,
                 gen_conv_dim=64, disc_conv_dim=64, gen_fc_dim=1024, disc_fc_dim=1024,
                 batch_size=64, sample_num=64, checkpoint_dir=None, sample_dir=None, model_dir=None,
                 learning_rate=0.0002, beta1=0.5):
        """
        Constructor
        :param sess: TensorFlow session
        :param name:
        :param input_width:
        :param input_height:
        :param output_width:
        :param output_height:
        :param y_dim: Dimension of dim for y. [None]
        :param z_dim: Dimension of dim for Z. [100]
        :param img_depth: Dimension of image color. For grayscale input, set to 1. [3]
        :param gen_conv_dim: Dimension of gen filters in first conv layer. [64]
        :param disc_conv_dim: Dimension of discrim filters in first conv layer. [64]
        :param gen_fc_dim: Dimension of gen units for for fully connected layer. [1024]
        :param disc_fc_dim: Dimension of discrim units for fully connected layer. [1024]
        :param batch_size: The size of batch. Should be specified before training.
        :param sample_num:
        :param checkpoint_dir:
        :param sample_dir:
        :param model_dir:
        """

        self.sess = sess
        self.name = name

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_width = input_width
        self.input_height = input_height
        self.output_width = output_width
        self.output_height = output_height

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gen_conv_dim = gen_conv_dim
        self.disc_conv_dim = disc_conv_dim

        self.gen_fc_dim = gen_fc_dim
        self.disc_fc_dim = disc_fc_dim

        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.model_dir = model_dir
        self.input_fname_pattern = "*.jpg"

        # mnist
        self.img_depth = 1
        self.grayscale = (self.img_depth == 1)

        # build graph
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
        image_dims = [self.input_height, self.input_width, self.img_depth]

        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')
        self.sample_inputs = tf.placeholder(tf.float32, [self.sample_num] + image_dims, name='sample_inputs')

        # generator
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = tf.summary.histogram("z", self.z)
        self.G = generator_mnist(self.z, self.y)

        # discriminator
        self.D, self.D_logits = discriminator_mnist(self.inputs, self.y, reuse=False)
        self.sampler = generator_mnist(self.z, self.y, reuse=True, is_training=False)
        self.D_, self.D_logits_ = discriminator_mnist(self.G, self.y, reuse=True)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.G_sum = tf.summary.histogram("G", self.G)

        # Loss
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        # trainable parameters
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')

        # optimizer
        self.opt_d = optimizer(loss=self.d_loss, var_list=self.d_vars, learning_rate=learning_rate, beta1=beta1)
        self.opt_g = optimizer(loss=self.g_loss, var_list=self.g_vars, learning_rate=learning_rate, beta1=beta1)

        # summary
        self.g_sum = tf.summary.merge([self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        # saver
        self.saver = tf.train.Saver()

    def update_discriminator(self, x_data, z_data, y_data):
        """
        DCGAN 구별자 학습 (gradient descent 1 step)
        :param x_data:
        :param z_data:
        :param y_data:
        :return:
        """
        loss_d, _ = self.sess.run([self.d_sum, self.opt_d],
                                  feed_dict={self.inputs: x_data, self.z: z_data, self.y: y_data})
        return loss_d

    def update_generator(self, z_data, y_data):
        """
        DCGAN 생성자 학습 (gradient descent 1 step)
        :param z_data:
        :param y_data:
        :return:
        """
        loss_g, _ = self.sess.run([self.g_sum, self.opt_g],
                                  feed_dict={self.z: z_data, self.y: y_data})
        return loss_g
