#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__title__ = 'data_generator_for_preposed_model'
__author__ = 'fangwudi'
__time__ = '18-11-20 14:18'

code is far away from bugs 
     ┏┓   ┏┓
    ┏┛┻━━━┛┻━┓
    ┃        ┃
    ┃ ┳┛  ┗┳ ┃
    ┃    ┻   ┃
    ┗━┓    ┏━┛
      ┃    ┗━━━━━┓
      ┃          ┣┓
      ┃          ┏┛
      ┗┓┓┏━━┳┓┏━━┛
       ┃┫┫  ┃┫┫
       ┗┻┛  ┗┻┛
with the god animal protecting
     
"""
from keras.preprocessing.image import *
# deal with image file is truncated error
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras import backend as K
import matplotlib.pyplot as plt
from skimage import transform as skimage_tf
import math, glob
import random
import numpy as np
from copy import deepcopy

from .match_produce import *


class MyImageDataGenerator(ImageDataGenerator):
    def my_get_random_transform(self, seed=None):
        """Generates random parameters for a transformation.

        # Arguments
            seed: Random seed.
            img_shape: Tuple of integers.
                Shape of the image that is transformed.

        # Returns
            A dictionary containing randomly chosen parameters describing the
            transformation.
        """
        if seed is not None:
            np.random.seed(seed)

        if self.rotation_range:
            theta = np.random.uniform(
                -self.rotation_range,
                self.rotation_range)
        else:
            theta = 0

        if self.height_shift_range:
            if np.max(np.abs(self.height_shift_range)) > 1:
                raise ValueError('height_shift_range should be -1~1')
            tx = np.random.uniform(-self.height_shift_range,
                                   self.height_shift_range)
        else:
            tx = 0

        if self.width_shift_range:
            if np.max(np.abs(self.width_shift_range)) > 1:
                raise ValueError('width_shift_range should be -1~1')
            ty = np.random.uniform(-self.width_shift_range,
                                   self.width_shift_range)
        else:
            ty = 0

        if self.shear_range:
            raise ValueError('shear_range not consist, so not implement')
            # shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(
                self.zoom_range[0],
                self.zoom_range[1],
                2)

        flip_horizontal = (np.random.random() < 0.5) * self.horizontal_flip
        flip_vertical = (np.random.random() < 0.5) * self.vertical_flip

        channel_shift_intensity = None
        if self.channel_shift_range != 0:
            channel_shift_intensity = np.random.uniform(-self.channel_shift_range, self.channel_shift_range)

        transform_parameters = {'theta': theta,
                                'tx': tx,
                                'ty': ty,
                                'shear': shear,
                                'zx': zx,
                                'zy': zy,
                                'flip_horizontal': flip_horizontal,
                                'flip_vertical':   flip_vertical,
                                'channel_shift_intensity': channel_shift_intensity}

        return transform_parameters

    def change_shift_parameters(self, transform_parameters, img_shape):
        result = deepcopy(transform_parameters)
        tx, ty = -1 * transform_parameters['tx'], -1 * transform_parameters['ty']
        zx, zy = 1 / transform_parameters['zx'], 1 / transform_parameters['zy']
        theta = 1 * transform_parameters['theta']
        shear = 1 * transform_parameters['shear']
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        tx *= img_shape[img_row_axis]
        ty *= img_shape[img_col_axis]
        result['tx'], result['ty'], result['theta'], result['zx'], result['zy'], result['shear'] = tx, ty, theta, zx, zy, shear
        return result

    def myflow_from_directory(self, directory,
                              target_size=(512, 512), color_mode='rgb',
                              batch_size=32, shuffle=True, seed=None,
                              save_to_dir=None,
                              save_prefix='',
                              save_format='png',
                              follow_links=True,
                              interpolation='nearest',
                              use_mask_ab=True,
                              heatmap_height=63,
                              heatmap_width=63,
                              gpu_num=2,
                              return_path=False,
                              simple_take_flag=False,
                              x_threshold=0.06, y_threshold=0.06):
        return MyDirectoryIterator(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            interpolation=interpolation,
            # self define
            use_mask_ab = use_mask_ab,
            heatmap_height=heatmap_height,
            heatmap_width=heatmap_width,
            gpu_num=gpu_num,
            return_path = return_path,
            simple_take_flag=simple_take_flag,
            x_threshold=x_threshold, y_threshold=y_threshold)


class MyDirectoryIterator(Iterator):
    """Iterator capable of reading images and annotation json from a directory on disk.

    # Arguments
        directory: Path to the directory to read images and annotation json from.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
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

    def __init__(self, directory,
                 image_data_generator,
                 target_size=(512, 512), 
                 color_mode='rgb',
                 batch_size=32, 
                 shuffle=True,
                 seed=None,
                 data_format=None, 
                 save_to_dir=None,
                 save_prefix='', 
                 save_format='png',
                 follow_links=False,
                 interpolation='nearest',
                 # self define
                 use_mask_ab=True,
                 heatmap_height=63,
                 heatmap_width=63,
                 gpu_num=2,
                 return_path=False,
                 simple_take_flag=False,
                 x_threshold=0.06, y_threshold=0.06):
        if data_format is None:
            data_format = K.image_data_format()
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,  '; expected "rgb" or "grayscale".')
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
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation
        # self define
        self.row_axis = image_data_generator.row_axis
        self.col_axis = image_data_generator.col_axis
        self.use_mask_ab = use_mask_ab
        self.heatmap_height = heatmap_height
        self.heatmap_width = heatmap_width
        self.return_path = return_path
        self.x_threshold = x_threshold
        self.y_threshold = y_threshold
        self.data_num, self.basic_directory_dict = gather_data(directory, simple_take_flag=simple_take_flag,
                                                               follow_links=follow_links)
        print("basic_directory numbers: {}".format(len(self.basic_directory_dict)))
        print('Found %d data.' % self.data_num)
        # decide if drop last batch when multi gpu
        last = self.data_num % batch_size
        if 0 < last < gpu_num:
            self.fill_last = True
        else:
            self.fill_last = False
        super(MyDirectoryIterator, self).__init__(self.data_num, batch_size, shuffle, seed)

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
        if idx == len(self) - 1 and self.fill_last:
            index_array = self.index_array[-self.batch_size:]
        else:
            index_array = self.index_array[self.batch_size * idx:self.batch_size * (idx + 1)]
        return self._get_batches_of_transformed_samples(index_array)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_img_a = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        batch_img_b = np.zeros_like(batch_img_a)
        batch_heatmap_a_change = np.zeros((len(index_array), self.heatmap_height, self.heatmap_width, 1), dtype=np.uint8)
        batch_mask_a_all = np.ones_like(batch_heatmap_a_change)
        batch_input_a_all = np.ones_like(batch_heatmap_a_change)
        grayscale = self.color_mode == 'grayscale'
        path_list = []
        # build batch of image data
        for i, j in enumerate(index_array):
            dirpath = random.choice(list(self.basic_directory_dict))
            filenames = self.basic_directory_dict[dirpath]
            a_filename, b_filename = random.choice(filenames), random.choice(filenames)
            a_path = os.path.join(dirpath, a_filename)
            b_path = os.path.join(dirpath, b_filename)
            # select more as a
            a_num = sum(read_json(a_path[:-4]+'.json').values())
            b_num = sum(read_json(b_path[:-4]+'.json').values())
            if a_num < b_num:
                temp = a_path
                a_path = b_path
                b_path = temp
            path_list.append([a_path, b_path])
            a_mp = read_json(a_path[:-4]+'_mp.json')
            b_mp = read_json(b_path[:-4] + '_mp.json')
            a_mp = dict([(k, str2float(v)) for k, v in a_mp.items() if len(v) > 0])
            b_mp = dict([(k, str2float(v)) for k, v in b_mp.items() if len(v) > 0])
            # calculate match
            a_change_s, _, a_same_s, _, _, _ = match_s(a_mp, b_mp, x_threshold=self.x_threshold, y_threshold=self.y_threshold)
            a_change = merge_double_list(a_change_s.values()) # a_sku changed related to b, 
            a_same = merge_double_list(a_same_s.values()) # a_sku remained in the same position related to b,
            random_seed = np.random.randint(0, 1000000)
            # load img
            img_a = load_img(os.path.join(a_path), grayscale=grayscale, target_size=self.target_size,
                             interpolation=self.interpolation)
            a = img_to_array(img_a, data_format=self.data_format)
            img_b = load_img(os.path.join(b_path), grayscale=grayscale, target_size=self.target_size,
                             interpolation=self.interpolation)
            b = img_to_array(img_b, data_format=self.data_format)
            # aug
            params = self.image_data_generator.my_get_random_transform(seed=random_seed)
            params_img = self.image_data_generator.change_shift_parameters(params, self.image_shape)
            a = self.image_data_generator.apply_transform(a, params_img)
            a = self.image_data_generator.standardize(a)
            b = self.image_data_generator.apply_transform(b, params_img)
            b = self.image_data_generator.standardize(b)
            
            if len(a_change):
                a_change = transform_point(a_change, params)
            if len(a_same):
                a_same = transform_point(a_same, params)
            a_change_heatmap = self.generate_heatmap_with_points(a_change)
            a_same_heatmap = self.generate_heatmap_with_points(a_same)
            
            batch_input_a_all[i] = np.logical_or(a_change_heatmap, a_same_heatmap).astype(np.uint8) # merging a_change and a_same
            batch_img_a[i] = a
            batch_img_b[i] = b
            batch_heatmap_a_change[i] = a_change_heatmap

        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img_a = array_to_img(batch_img_a[i], self.data_format, scale=True)
                fname_a = '{prefix}_{index}_{hash}.{format}'.format(prefix='a' + self.save_prefix,
                                                                    index=j,
                                                                    hash=np.random.randint(1e7),
                                                                    format=self.save_format)
                img_a.save(os.path.join(self.save_to_dir, fname_a))
                img_b = array_to_img(batch_img_b[i], self.data_format, scale=True)
                fname_b = '{prefix}_{index}_{hash}.{format}'.format(prefix='b' + self.save_prefix,
                                                                    index=j,
                                                                    hash=np.random.randint(1e7),
                                                                    format=self.save_format)
                img_b.save(os.path.join(self.save_to_dir, fname_b))
        if not self.return_path:
            return [batch_img_a, batch_img_b, batch_input_a_all], batch_heatmap_a_change
        else:
            return [batch_img_a, batch_img_b, batch_input_a_all], batch_heatmap_a_change, path_list

    def generate_heatmap_with_points(self, point_list):
        heatmap = np.zeros((self.heatmap_height, self.heatmap_width), dtype=np.uint8)
        for point in point_list:
            h, w = point[1], point[0]
            if 0 <= h < 1 and 0 <= w < 1:
                h, w = int(h * self.heatmap_height), int(w * self.heatmap_width)
                heatmap[h][w] = 1
        return heatmap[:, :, np.newaxis]

    @staticmethod
    def read_file(file_name):
        f = open(file_name)
        r = f.read()
        f.close()
        return r

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



def gather_data(directory, simple_take_flag = False, follow_links=True):
    def _recursive_list(subpath):
        return os.walk(subpath, followlinks=follow_links)
    
    data_num = 0
    basic_directory_dict = {}
    
    for x_dirpath, _, third_filenames in _recursive_list(directory):
        if len(third_filenames) != 0:
            valid_filenames = []
            max_num  = 0
            first_found = False
            sorted_filenames = sorted(glob.glob(os.path.join(x_dirpath, '*.jpg')))
            if len(sorted_filenames) > 1:
                if simple_take_flag:
                    for x_filename in sorted_filenames:
                        num = read_json(x_filename[:-4] + '.json')
                        now_num = sum(num.values())
                        if not first_found and max_num  < now_num:
                            max_num = now_num
                            first_filename = x_filename
                            continue
                        else:
                            first_found = True
                            data_path = x_filename[:-4] + '_mp.json'
                            if read_json(data_path) is None:
                                print(data_path)
                            else:
                                valid_filenames.append(x_filename.split(x_dirpath)[-1][1:])
                    valid_filenames.append(first_filename.split(x_dirpath)[-1][1:])
                    if max_num == 0:
                            print('{} the first image is empty.'.format(first_filename))
                                      
                else:
                    for x_filename in sorted_filenames:
                        num = read_json( x_filename[:-4] + '.json')
                        data_path = x_filename[:-4] + '_mp.json'
                        if read_json(data_path) is None:
                            print(data_path)
                        else:
                            valid_filenames.append(x_filename.split(x_dirpath)[-1][1:])
                data_num += len(valid_filenames)
                basic_directory_dict[x_dirpath] = valid_filenames
    return data_num, basic_directory_dict


def merge_double_list(double_list):
    res = []
    for i in double_list:
        if len(i):
            for j in i:
                res.append(j)
    return res


def read_json(file_name):
    f = open(file_name)
    r = f.read()
    f.close()
    if not r:
        return None
    j = json.loads(r)
    return j


def transform_point(point_list, transform_parameters):
    scale_x, scale_y = transform_parameters['zy'], transform_parameters['zx']
    translate_x_px, translate_y_px = transform_parameters['ty'], transform_parameters['tx']
    rotate = transform_parameters['theta']
    shear = transform_parameters['shear']
    flip_horizontal = transform_parameters.get('flip_horizontal', False)
    flip_vertical = transform_parameters.get('flip_vertical', False)
    if scale_x != 1.0 or scale_y != 1.0 or translate_x_px != 0 or translate_y_px != 0 or rotate != 0 \
            or shear != 0:
        matrix_to_topleft = skimage_tf.SimilarityTransform(translation=[-0.5, -0.5])
        matrix_transforms = skimage_tf.AffineTransform(
            scale=(scale_x, scale_y),
            translation=(translate_x_px, translate_y_px),
            rotation=math.radians(rotate),
            shear=math.radians(shear)
        )
        matrix_to_center = skimage_tf.SimilarityTransform(translation=[0.5, 0.5])
        matrix = (matrix_to_topleft + matrix_transforms + matrix_to_center)
        point_list = skimage_tf.matrix_transform(point_list, matrix.params)
    if flip_horizontal or flip_vertical:
        matrix_to_topleft = skimage_tf.SimilarityTransform(translation=[-0.5, -0.5])
        point_list = skimage_tf.matrix_transform(point_list, matrix_to_topleft.params)
        if flip_horizontal:
            point_list = [(-x, y) for x, y in point_list]
        if flip_vertical:
            point_list = [(x, -y) for x, y in point_list]
        matrix_to_center = skimage_tf.SimilarityTransform(translation=[0.5, 0.5])
        point_list = skimage_tf.matrix_transform(point_list, matrix_to_center.params)
    return point_list


def display_images(images, cols=5, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interporlation to use for display.
    """
    plt.figure(figsize=(18, 18 * cols))
    i = 1
    for one_image in images:
        plt.subplot(1, cols, i)
        plt.axis('off')
        plt.imshow(one_image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()
