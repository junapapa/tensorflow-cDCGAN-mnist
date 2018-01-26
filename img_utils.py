"""
    @ file : img_utils.py
    @ brief

    @ author : Younghyun Lee <yhlee109@gmail.com>
    @ date : 2018.01.25
    @ version : 1.0
"""
import numpy as np
import scipy.misc
import math
from time import gmtime, strftime
import random

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def imsave(images, size, path):
    """

    :param images:
    :param size:
    :param path:
    :return:
    """
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)


def inverse_transform(images):
    """

    :param images:
    :return:
    """
    return (images+1.)/2.


def merge(images, size):
    """

    :param images:
    :param size:
    :return:
    """
    h, w = images.shape[1], images.shape[2]

    if images.shape[3] in (3, 4):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image

        return img

    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]

        return img

    else:
        raise ValueError('in merge(images,size) images parameter must have dimensions: HxW or HxWx3 or HxWx4')


def visualize_mnist(model, option):
    """

    :param model:
    :param option:
    :return:
    """

    image_frame_dim = int(math.ceil(model.batch_size**.5))

    if option == 0:
        """
        
        """
        z_sample = np.random.uniform(-0.8, 0.8, size=(model.batch_size, model.z_dim))

        y = np.random.choice(10, model.batch_size)
        y_one_hot = np.zeros((model.batch_size, 10))
        y_one_hot[np.arange(model.batch_size), y] = 1

        samples = model.sess.run(model.sampler, feed_dict={model.z: z_sample, model.y: y_one_hot})
        save_images(samples, [image_frame_dim, image_frame_dim],
                    './samples/test_all_classes_%s.png' % strftime("%Y%m%d%H%M%S", gmtime()))

        # each class
        n_styles = 10   # must be less than or equal to model.batch_size

        np.random.seed()
        si = np.random.choice(model.batch_size, n_styles)

        all_samples = []
        for l_idx in range(10):
            y = np.zeros(model.batch_size, dtype=np.int64) + l_idx
            y_one_hot = np.zeros((model.batch_size, 10))
            y_one_hot[np.arange(model.batch_size), y] = 1

            samples = model.sess.run(model.sampler, feed_dict={model.z: z_sample, model.y: y_one_hot})
            save_images(samples, [image_frame_dim, image_frame_dim],
                        './samples/test_class_%d_%s.png' % (l_idx, strftime("%Y%m%d%H%M%S", gmtime())))

            samples = samples[si, :, :, :]

            if l_idx == 0:
                all_samples = samples
            else:
                all_samples = np.concatenate((all_samples, samples), axis=0)

        # save meged images to check style-consistency
        canvas = np.zeros_like(all_samples)
        for s in range(n_styles):
            for c in range(10):
                canvas[s*10+c, :, :, :] = all_samples[c*n_styles+s, :, :, :]

        save_images(canvas, [n_styles, 10],
                    './samples/test_all_classes_style_by_style_%s.png' % (strftime("%Y%m%d%H%M%S", gmtime())))

    elif option == 1:
        """
        """
        values = np.arange(0, 1, 1./model.batch_size)

        for idx in range(100):
            print(" [*] %d" % idx)

            z_sample = np.zeros([model.batch_size, model.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            y = np.random.choice(10, model.batch_size)
            y_one_hot = np.zeros((model.batch_size, 10))
            y_one_hot[np.arange(model.batch_size), y] = 1

            samples = model.sess.run(model.sampler, feed_dict={model.z: z_sample, model.y: y_one_hot})

            save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_arange_%s.png' % (idx))

    elif option == 2:
        """
        """
        values = np.arange(0, 1, 1./model.batch_size)
        for idx in [random.randint(0, 99) for _ in range(100)]:
            print(" [*] %d" % idx)
            z = np.random.uniform(-0.2, 0.2, size=model.z_dim)
            z_sample = np.tile(z, (model.batch_size, 1))

            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]


            y = np.random.choice(10, model.batch_size)
            y_one_hot = np.zeros((model.batch_size, 10))
            y_one_hot[np.arange(model.batch_size), y] = 1

            samples = model.sess.run(model.sampler, feed_dict={model.z: z_sample, model.y: y_one_hot})

            try:
                pass
                # make_gif(samples, './samples/test_gif_%s.gif' % (idx))

            except:
                save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_%s.png' % strftime("%Y%m%d%H%M%S", gmtime()))

    elif option == 3:
        """
        values = np.arange(0, 1, 1./model.batch_size)
        for idx in range(100):
            print(" [*] %d" % idx)
            z_sample = np.zeros([model.batch_size, model.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            y = np.random.choice(10, model.batch_size)
            y_one_hot = np.zeros((model.batch_size, 10))
            y_one_hot[np.arange(model.batch_size), y] = 1

            samples = model.sess.run(model.sampler, feed_dict={model.z: z_sample, model.y: y_one_hot})

            make_gif(samples, './samples/test_gif_%s.gif' % (idx))
        """
    elif option == 4:
        """
        image_set = []
        values = np.arange(0, 1, 1./model.batch_size)

        for idx in range(100):
            print(" [*] %d" % idx)
            z_sample = np.zeros([model.batch_size, model.z_dim])
            for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

            y = np.random.choice(10, model.batch_size)
            y_one_hot = np.zeros((model.batch_size, 10))
            y_one_hot[np.arange(model.batch_size), y] = 1

            image_set.append(model.sess.run(model.sampler, feed_dict={model.z: z_sample, model.y: y_one_hot}))

            make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

        new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10])
                         for idx in range(64) + range(63, -1, -1)]

        make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)
        """

"""
def make_gif(images, fname, duration=2, true_image=False):

    import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images)/duration*t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)
"""