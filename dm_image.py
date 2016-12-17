import numpy as np
from numpy.random import RandomState
from os import path
from keras.preprocessing.image import ImageDataGenerator, Iterator
import keras.backend as K
import cv2
from meta import UNIMAGED_INT

class DMImgListIterator(Iterator):
    '''An iterator for a flatten image list
    '''

    def __init__(self, img_list, lab_list, image_data_generator,
                 target_size=(1152, 896), gs_255=False, dim_ordering='default',
                 class_mode='binary',
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        '''DM image iterator
        Args:
            target_size (tuple of int): (width, height).
        '''
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.gs_255 = gs_255
        self.dim_ordering = dim_ordering
        # Always gray-scale.
        if self.dim_ordering == 'tf':
            self.image_shape = self.target_size + (1,)
        else:
            self.image_shape = (1,) + self.target_size
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        # Convert flattened image list.
        self.nb_sample = len(img_list)
        self.nb_class = 2
        self.filenames = list(img_list)
        self.classes = np.array(lab_list)
        nb_pos = np.sum(self.classes == 1)
        nb_neg = np.sum(self.classes == 0)
        print('There are %d cancer cases and %d normal cases.' % (nb_pos, nb_neg))

        super(DMImgListIterator, self).__init__(
            self.nb_sample, batch_size, shuffle, seed)


    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype='float32')
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            if self.gs_255:
                img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
            if self.target_size != img.shape:
                img = cv2.resize(
                    img, dsize=(self.target_size[1], self.target_size[0]), 
                    interpolation=cv2.INTER_CUBIC)
            img = img.astype('float32')
            # Always have one channel.
            if self.dim_ordering == 'th':
                x = img.reshape((1, img.shape[0], img.shape[1]))
            else:
                x = img.reshape((img.shape[0], img.shape[1], 1))
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img = batch_x[i]
                if self.dim_ordering == 'th':
                    img = img.reshape((img.shape[1], img.shape[2]))
                else:
                    img = img.reshape((img.shape[0], img.shape[1]))
                cv2.imwrite(path.join(self.save_to_dir, fname), img)
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype('float32')
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.nb_class), dtype='float32')
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y


class DMExamListIterator(Iterator):
    '''An iterator for a flatten exam list
    '''

    def __init__(self, exam_list, image_data_generator,
                 target_size=(1152, 896), gs_255=False, dim_ordering='default',
                 class_mode='binary',
                 batch_size=16, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.gs_255 = gs_255
        self.dim_ordering = dim_ordering
        # Always gray-scale. Two inputs: CC and MLO.
        if self.dim_ordering == 'tf':
            self.image_shape = self.target_size + (1,)
        else:
            self.image_shape = (1,) + self.target_size
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.seed = seed
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        # Convert exam list.
        self.exam_list = exam_list
        self.nb_exam = len(exam_list)
        self.nb_class = 2
        # For each exam: 0 => subj id, 1 => exam idx, 2 => exam dat.
        self.classes = [ (e[2]['L']['cancer'], e[2]['R']['cancer']) 
                         for e in exam_list ]
        self.classes = np.array(self.classes)
        print('For left breasts, normal=%d, cancer=%d, unimaged=%d.' % 
            (np.sum(self.classes[:, 0] == 0), 
             np.sum(self.classes[:, 0] == 1), 
             np.sum(self.classes[:, 0] == UNIMAGED_INT)))
        print('For right breasts, normal=%d, cancer=%d, unimaged=%d.' % 
            (np.sum(self.classes[:, 1] == 0), 
             np.sum(self.classes[:, 1] == 1), 
             np.sum(self.classes[:, 1] == UNIMAGED_INT)))

        super(DMExamListIterator, self).__init__(
            self.nb_exam, batch_size, shuffle, seed)


    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
            # Obtain the current random state to draw images randomly.
            current_batch_rs = RandomState(self.seed + self.total_batches_seen)
        # The transformation of images is not under thread lock so it can be done in parallel
        nb_unimaged = np.sum(self.classes[index_array, :] == UNIMAGED_INT)
        # batch size measures the number of exams, an exam has two breasts.
        current_batch_size = current_batch_size*2 - nb_unimaged
        batch_x_cc = np.zeros( (current_batch_size,) + self.image_shape, dtype='float32' )
        batch_x_mlo = np.zeros( (current_batch_size,) + self.image_shape, dtype='float32' )

        def rand_draw_img(img_df):
            '''Randomly read an image when there is repeated imagings
            '''
            fname = img_df['filename'].sample(1, random_state=current_batch_rs).iloc[0]
            if self.gs_255:
                img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
            if self.target_size != img.shape:
                img = cv2.resize(
                    img, dsize=(self.target_size[1], self.target_size[0]), 
                    interpolation=cv2.INTER_CUBIC)
            return img.astype('float32')

        def read_transform_breast_imgs(breast_dat):
            '''Read and transform the images for both views for a breast
            '''
            #!!! if a view is missing, use a zero-filled 2D array.
            #!!! this may need to be changed depending on the deep learning design.
            if breast_dat['CC'] is None:
                img_cc = np.zeros(self.target_size, dtype='float32')
            else:
                img_cc = rand_draw_img(breast_dat['CC'])
            if breast_dat['MLO'] is None:
                img_mlo = np.zeros(self.target_size, dtype='float32')
            else:
                img_mlo = rand_draw_img(breast_dat['MLO'])
            # Always have one channel.
            if self.dim_ordering == 'th':
                img_cc = img_cc.reshape((1, img_cc.shape[0], img_cc.shape[1]))
                img_mlo = img_mlo.reshape((1, img_mlo.shape[0], img_mlo.shape[1]))
            else:
                img_cc = img_cc.reshape((img_cc.shape[0], img_cc.shape[1], 1))
                img_mlo = img_mlo.reshape((img_mlo.shape[0], img_mlo.shape[1], 1))
            img_cc = self.image_data_generator.random_transform(img_cc)
            img_cc = self.image_data_generator.standardize(img_cc)
            img_mlo = self.image_data_generator.random_transform(img_mlo)
            img_mlo = self.image_data_generator.standardize(img_mlo)

            return (img_cc, img_mlo)
            # if self.dim_ordering == 'th':
            #     return np.stack((img_cc, img_mlo), axis=0)
            # else:
            #     return np.stack((img_cc, img_mlo), axis=-1)
            
        # build batch of image data
        adv = 0
        for i in index_array:
            exam_dat = self.exam_list[i][2]
            if not self.classes[i, 0] == UNIMAGED_INT:
                img_cc, img_mlo = read_transform_breast_imgs(exam_dat['L'])
                batch_x_cc[adv] = img_cc
                batch_x_mlo[adv] = img_mlo
                adv += 1
            if not self.classes[i, 1] == UNIMAGED_INT:
                img_cc, img_mlo = read_transform_breast_imgs(exam_dat['R'])
                batch_x_cc[adv] = img_cc
                batch_x_mlo[adv] = img_mlo
                adv += 1

        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                fname_base = '{prefix}_{index}_{hash}'.format(prefix=self.save_prefix,
                                                              index=current_index + i,
                                                              hash=np.random.randint(1e4))
                fname_cc = fname_base + '_cc.' + self.save_format
                fname_mlo = fname_base + '_mlo.' + self.save_format
                img_cc = batch_x_cc[i]
                img_mlo = batch_x_mlo[i]
                if self.dim_ordering == 'th':
                    img_cc = img_cc.reshape((img_cc.shape[1], img_cc.shape[2]))
                    img_mlo = img_mlo.reshape((img_mlo.shape[1], img_mlo.shape[2]))
                else:
                    img_cc = img_cc.reshape((img_cc.shape[0], img_cc.shape[1]))
                    img_mlo = img_mlo.reshape((img_mlo.shape[0], img_mlo.shape[1]))
                cv2.imwrite(path.join(self.save_to_dir, fname_cc), img_cc)
                cv2.imwrite(path.join(self.save_to_dir, fname_mlo), img_mlo)
        
        # build batch of labels
        flat_classes = self.classes[index_array, :].ravel()  # [L, R, L, R, ...]
        flat_classes = flat_classes[flat_classes != UNIMAGED_INT]
        if self.class_mode == 'sparse':
            batch_y = flat_classes
        elif self.class_mode == 'binary':
            batch_y = flat_classes.astype('float32')
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.nb_class), dtype='float32')
            for i, label in enumerate(flat_classes):
                batch_y[i, label] = 1.
        else:
            return [batch_x_cc, batch_x_mlo]
        return [batch_x_cc, batch_x_mlo], batch_y


class DMImageDataGenerator(ImageDataGenerator):
    '''Image data generator for digital mammography
    '''

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 dim_ordering='default'):

        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        super(DMImageDataGenerator, self).__init__(
            featurewise_center=featurewise_center,
            samplewise_center=samplewise_center,
            featurewise_std_normalization=featurewise_std_normalization,
            samplewise_std_normalization=samplewise_std_normalization,
            zca_whitening=zca_whitening,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            channel_shift_range=channel_shift_range,
            fill_mode=fill_mode,
            cval=cval,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            rescale=rescale,
            dim_ordering=dim_ordering)


    def flow_from_img_list(self, img_list, lab_list, 
                           target_size=(1152, 896), gs_255=False, class_mode='binary',
                           batch_size=32, shuffle=True, seed=None,
                           save_to_dir=None, save_prefix='', save_format='jpeg'):
        return DMImgListIterator(
            img_list, lab_list, self, 
            target_size=target_size, gs_255=gs_255, class_mode=class_mode,
            dim_ordering=self.dim_ordering,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)


    def flow_from_exam_list(self, exam_list, 
                           target_size=(1152, 896), gs_255=False, class_mode='binary',
                           batch_size=16, shuffle=True, seed=None,
                           save_to_dir=None, save_prefix='', save_format='jpeg'):
        return DMExamListIterator(
            exam_list, self, 
            target_size=target_size, gs_255=gs_255, class_mode=class_mode,
            dim_ordering=self.dim_ordering,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

















