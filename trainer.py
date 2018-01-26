"""
    @ file : trainer.py
    @ brief

    @ author : Younghyun Lee <yhlee109@gmail.com>
    @ date : 2018.01.25
    @ version : 1.0
"""
import os
import numpy as np
import tensorflow as tf
import time
import re
from img_utils import *


def optimizer(loss, var_list, learning_rate=0.0002, beta1=0.5):
    """
    Adam Optimizer
    :param loss:
    :param var_list:
    :param learning_rate:
    :param beta1:
    :return:
    """
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(loss=loss, var_list=var_list)

    return opt


def run_train(model, train_set, training_epochs, flag_checkpoint, checkpoint_counter=0):
    """

    :param model:
    :param training_epochs:
    :return:
    """

    sample_z = np.random.uniform(-1, 1, size=(model.sample_num, model.z_dim))

    sample_inputs = train_set.train.images[0:model.sample_num]
    sample_inputs = np.reshape(sample_inputs, (-1, 28, 28, 1))
    sample_labels = train_set.train.labels[0:model.sample_num]

    counter = 1
    start_time = time.time()

    if flag_checkpoint:
        counter = checkpoint_counter
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")

    # training_loop
    print('\n===== Start : DCGAN training =====\n')
    for epoch in range(training_epochs):

        total_batch = int(train_set.train.images.shape[0] / model.batch_size)

        for idx in range(total_batch):
            batch_images = train_set.train.images[idx*model.batch_size:(idx+1)*model.batch_size]
            batch_images = np.reshape(batch_images, (-1, 28, 28, 1))
            batch_labels = train_set.train.labels[idx*model.batch_size:(idx+1)*model.batch_size]

            batch_z = np.random.uniform(-1, 1, [model.batch_size, model.z_dim]).astype(np.float32)

            # update discriminator
            summary_str = model.update_discriminator(x_data=batch_images, z_data=batch_z, y_data=batch_labels)
            model.writer.add_summary(summary_str, counter)

            # update generator
            summary_str = model.update_generator(z_data=batch_z, y_data=batch_labels)
            model.writer.add_summary(summary_str, counter)

            # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
            summary_str = model.update_generator(z_data=batch_z, y_data=batch_labels)
            model.writer.add_summary(summary_str, counter)

            errD_fake = model.sess.run(model.d_loss_fake, {model.z: batch_z, model.y: batch_labels})
            errD_real = model.sess.run(model.d_loss_real, {model.inputs: batch_images, model.y: batch_labels})
            errG = model.sess.run(model.g_loss, {model.z: batch_z, model.y: batch_labels})

            counter += 1
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                  % (epoch, idx, total_batch, time.time() - start_time, errD_fake + errD_real, errG))

            if np.mod(counter, 100) == 1:

                samples, d_loss, g_loss = model.sess.run([model.sampler, model.d_loss, model.g_loss],
                                                         feed_dict={model.z: sample_z,
                                                                    model.inputs: sample_inputs,
                                                                    model.y: sample_labels
                                                                    }
                                                         )
                manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
                manifold_w = int(np.floor(np.sqrt(samples.shape[0])))

                save_images(samples, [manifold_h, manifold_w],
                            './{}/train_{:02d}_{:04d}.png'.format(model.sample_dir, epoch, idx))

                print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

            if np.mod(counter, 500) == 2:
                save_checkpoint(model, counter)


def load_checkpoint(model):
    """

    :param model:
    :return:
    """

    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(model.checkpoint_dir, model.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        model.saver.restore(model.sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print(" [*] Failed to find a checkpoint")
        return False, 0


def save_checkpoint(model, step):
    """

    :param model:
    :param step:
    :return:
    """
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(model.checkpoint_dir, model.model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    model.saver.save(model.sess, os.path.join(checkpoint_dir, model_name), global_step=step)
