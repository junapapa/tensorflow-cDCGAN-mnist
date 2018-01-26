"""
    @ file : main_mnist.py
    @ brief

    @ author : Younghyun Lee <yhlee109@gmail.com>
    @ date : 2018.01.25
    @ version : 1.0
"""
import os.path
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from trainer import *

from tensorflow.examples.tutorials.mnist import input_data
from dcgan import DCGAN


def main():
    """
    Main Function
    :return:
    """

    flag_train = True

    # parameters
    training_epochs = 25
    batch_size = 64

    learning_rate = 0.0002
    beta1 = 0.5

    input_width = 28
    input_height = 28

    output_width = 28
    output_height = 28

    # directories
    checkpoint_dir = "checkpoints"
    model_dir = "mnist"
    sample_dir = "samples"

    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    if not os.path.isdir(sample_dir):
        os.mkdir(sample_dir)

    # Load MNIST
    """ Check out https://www.tensorflow.org/get_started/mnist/beginners for
        more information about the mnist dataset 
    """
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    #train_set = mnist.train.images      # normalization; range: 0 ~ 1
    #train_set = (mnist.train.images - 0.5) / 0.5  # normalization; range: -1 ~ 1

    # Launch graph
    sess = tf.Session()
    dcgan = DCGAN(sess=sess, name='dcgan_mnist',
                  input_width=input_width, input_height=input_height,
                  output_height=output_height, output_width=output_width,
                  y_dim=10, z_dim=100, img_depth=1,
                  batch_size=batch_size, sample_num=batch_size,
                  checkpoint_dir=checkpoint_dir, sample_dir=sample_dir, model_dir=model_dir,
                  learning_rate=learning_rate, beta1=beta1)

    sess.run(tf.global_variables_initializer())

    # show_all_variables()
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

    # load trained model
    flag_checkpoint, counter = load_checkpoint(dcgan)

    if counter == 0:
        run_train(model=dcgan,
                  train_set=mnist,
                  training_epochs=training_epochs,
                  flag_checkpoint=flag_checkpoint,
                  checkpoint_counter=counter)

    visualize_mnist(model=dcgan, option=0)


if __name__ == '__main__':
  main()
