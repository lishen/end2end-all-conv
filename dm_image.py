import numpy as np
from os import path
from keras.preprocessing.image import ImageDataGenerator, Iterator
import keras.backend as K
import cv2


class DMImgListIterator(Iterator):

    def __init__(self, img_list, lab_list, image_data_generator,
                 target_size=(256, 256), dim_ordering='default',
                 class_mode='sparse',
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
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
        batch_x = np.zeros((current_batch_size,) + self.image_shape)
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, dsize=self.target_size, 
                             interpolation=cv2.INTER_CUBIC)
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


class DMImageDataGenerator(ImageDataGenerator):

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
                           target_size=(256, 256), class_mode='sparse',
                           batch_size=32, shuffle=True, seed=None,
                           save_to_dir=None, save_prefix='', save_format='jpeg'):
        return DMImgListIterator(
            img_list, lab_list, self, 
            target_size=target_size, class_mode=class_mode,
            dim_ordering=self.dim_ordering,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)


    def flow_from_exam_list():
        pass
















