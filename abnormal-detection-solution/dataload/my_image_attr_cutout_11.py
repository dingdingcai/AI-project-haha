#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__title__ = "my_image"
__author__ = 'fangwudi'
__email__ = 'fangwudi@foxmail.com'
__time__ = '17-10-24 上午10:30'
__abstract__ = 'my_image, '

       code is far away from bugs 
             ┏┓   ┏┓
            ┏┛┻━━━┛┻━┓
            ┃   ━    ┃
            ┃ ┳┛  ┗┳ ┃
            ┃    ┻   ┃
            ┗━┓    ┏━┛
              ┃    ┗━━━━━┓
              ┃ 神兽保佑  ┣┓
              ┃ 永无BUG!  ┏┛
              ┗┓┓┏━━┳┓┏━━┛
               ┃┫┫  ┃┫┫
               ┗┻┛  ┗┻┛
     with the god animal protecting
     
"""
from keras.preprocessing.image import *
import numpy as np
import multiprocessing
from functools import partial
import os, re
import keras.backend as K

class MyImageDataGenerator(ImageDataGenerator):
    def myflow_from_directory(self, directory,
                              target_size=(256, 256), color_mode='rgb',
                              classes=None, class_mode='categorical',
                              batch_size=32, shuffle=True, seed=None,
                              save_to_dir=None, save_prefix='',
                              save_format='png', follow_links=False,
                              interpolation='nearest',
                              my_horizontal_flip=False,my_cutout=False):
        
        return MyDirectoryIterator(
             directory, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_prefix=save_prefix,
            save_format=save_format, follow_links=follow_links,
            interpolation=interpolation,my_horizontal_flip=my_horizontal_flip,my_cutout=my_cutout)


class MyDirectoryIterator(Iterator):
    """Iterator capable of reading images from a directory on disk.

    # Arguments
        directory: Path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
        classes: Optional list of strings, names of subdirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are images identical to input images (mainly
                used to work with autoencoders),
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
    """

    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None, save_to_dir=None,
                 save_prefix='', save_format='png',
                 follow_links=False, interpolation='nearest',my_horizontal_flip=False,my_cutout=False):
        if data_format is None:
            data_format = K.image_data_format()
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             ' or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation
        self.my_horizontal_flip = my_horizontal_flip
        self.my_cutout = my_cutout

        # first, count the number of samples and classes
        self.samples = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        pool = multiprocessing.pool.ThreadPool()
        function_partial = partial(_count_valid_files_in_directory,
                                   follow_links=follow_links)
        self.samples = sum(pool.map(function_partial,
                                    (os.path.join(directory, subdir)
                                     for subdir in classes)))

        print('Found %d samples belonging to %d classes.' % (self.samples, self.num_classes))

        # second, build an index of the images in the different class subfolders
        results = []

        self.filenames = []
        self.classes = np.zeros((self.samples,), dtype='int32')
        i = 0
        for dirpath in (os.path.join(directory, subdir) for subdir in classes):
            results.append(pool.apply_async(_list_valid_filenames_in_directory,
                                            (dirpath, self.class_indices, follow_links)))
        for res in results:
            classes, filenames = res.get()
            self.classes[i:i + len(classes)] = classes
            self.filenames += filenames
            i += len(classes)
        pool.close()
        pool.join()
        super(MyDirectoryIterator, self).__init__(self.samples, batch_size, shuffle, seed)
    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        # deal with multi gpu 0 batch error
        if idx == len(self) - 1:
            idx = 0
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batches_of_transformed_samples(index_array)


    def _flip_axis(self, x, axis):
        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)
        return x
    def _get_batches_of_transformed_samples(self, index_array):
        # generate image 
        batch_img = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())

        batch_attr = np.zeros((len(index_array), 7), dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
        # the name of image
            fname = self.filenames[j]

            
            # print(batch_attr.shape)
            # print(batch_attr[i])
            # break
            img_dat = load_img(os.path.join(self.directory, fname),
                             grayscale=grayscale,
                             target_size=self.target_size,
                             interpolation=self.interpolation)
            img = img_to_array(img_dat, data_format=self.data_format)

            if 'collections' in self.directory:
                ab_info = re.split('[_.]', fname)[-3]
            else:
                ab_info = re.split('[_.]', fname)[-2]

            if self.my_cutout:
                if np.random.random() < 0.5:
                    mask_pad = 10
                    wid_img = img.shape[0]
                    hei_img = img.shape[1]
                    mask_size = mask_pad * 2 + 1
                    mask = np.ones((mask_size, mask_size, 3)) * 127.5
                    pad_img = np.zeros((wid_img + 2 * mask_pad, hei_img + 2 * mask_pad, 3))
                    pad_img[mask_pad : wid_img + mask_pad, mask_pad : hei_img + mask_pad, :] = img
                    for _ in range(5):
                        xa = np.random.randint(mask_pad, wid_img + mask_pad)
                        ya = np.random.randint(mask_pad, hei_img + mask_pad)
                        pad_img[xa - mask_pad : xa + mask_pad + 1, ya - mask_pad : ya + mask_pad + 1, :] = mask
                    img = pad_img[mask_pad : wid_img + mask_pad, mask_pad : hei_img + mask_pad, :]

            if self.my_horizontal_flip:
                if np.random.random() < 0.5:
                    img = self._flip_axis(img, 1)
                    ab_info = ab_info[::-1]

            # import pdb;pdb.set_trace()
            if len(ab_info) == 7:
                batch_attr[i] = [int(c) for c in ab_info]
            else:
                print(fname)
                raise('Error!')
#                continue


            img = self.image_data_generator.random_transform(img)
            img = self.image_data_generator.standardize(img)
            batch_img[i] = img
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_img[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=j,
                                                                  hash=np.random.randint(1e7),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # concat img a and b
        batch_x = batch_img
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':

            batch_y = self.classes[index_array].astype(K.floatx())
            batch_y = [batch_y, batch_attr]
            
            # print('class:')
            # print(self.classes[index_array])
            # print('batch_y:')
            # print(batch_y)
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(index_array), self.num_classes), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        # print(batch_x, batch_y)
        return batch_x, batch_y

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


def _count_valid_files_in_directory(directory, follow_links):
    """Count files with extension in `white_list_formats` contained in a directory.

    # Arguments
        directory: absolute path to the directory containing files to be counted
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.

    # Returns
        the count of files with extension in `white_list_formats` contained in
        the directory.
    """
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=follow_links), key=lambda tpl: tpl[0])

    samples = 0
    for root, _, files in _recursive_list(directory):
        for fname in files:
            if os.path.exists(os.path.join(root, fname)):
                samples += 1
    return samples

def _list_valid_filenames_in_directory(directory, class_indices, follow_links):
    """List paths of files in `subdir` relative from `directory` whose extensions are in `white_list_formats`.

    # Arguments
        directory: absolute path to a directory containing the files to list.
            The directory name is used as class label and must be a key of `class_indices`.
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
        class_indices: dictionary mapping a class name to its index.

    # Returns
        classes: a list of class indices
        filenames: the path of valid files in `directory`, relative from
            `directory`'s parent (e.g., if `directory` is "dataset/class1",
            the filenames will be ["class1/file1.jpg", "class1/file2.jpg", ...]).
    """
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=follow_links), key=lambda tpl: tpl[0])

    classes = []
    filenames = []
    subdir = os.path.basename(directory)
    basedir = os.path.dirname(directory)
    for root, _, files in _recursive_list(directory):
        for fname in sorted(files):
            if os.path.exists(os.path.join(root, fname)):
                classes.append(class_indices[subdir])
                # add filename relative to directory
                absolute_path = os.path.join(root, fname)
                filenames.append(os.path.relpath(absolute_path, basedir))
    return classes, filenames
