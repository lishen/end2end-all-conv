import numpy as np
from numpy.random import RandomState
from os import path
from keras.preprocessing.image import ImageDataGenerator, Iterator
from keras.utils.np_utils import to_categorical
import keras.backend as K
import cv2
import dicom


def index_balancer(index_array, classes, ratio, rng):
    '''Balance an index array according to desired neg:pos ratio
    '''
    current_batch_size = len(index_array)
    pos_weight = len(classes) / (np.sum(classes==1) + 1e-7)
    neg_weight = len(classes) / (np.sum(classes==0) + 1e-7)
    neg_weight *= ratio
    probs = np.zeros(current_batch_size)
    probs[classes==1] = pos_weight
    probs[classes==0] = neg_weight
    probs /= probs.sum()
    index_array = rng.choice(index_array, current_batch_size, p=probs)
    index_array.sort()  # can avoid repeated img reading.
    return index_array


def read_resize_img(fname, target_size, gs_255=False):
    '''Read an image (.png, .jpg, .dcm) and resize it to target size.
    '''
    if path.splitext(fname)[1] == '.dcm':
        img = dicom.read_file(fname).pixel_array
    else:
        if gs_255:
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    if target_size != img.shape:
        img = cv2.resize(
            img, dsize=(target_size[1], target_size[0]), 
            interpolation=cv2.INTER_CUBIC)
    img = img.astype('float32')
    return img


class DMImgListIterator(Iterator):
    '''An iterator for a flatten image list
    '''

    def __init__(self, img_list, lab_list, image_data_generator,
                 target_size=(1152, 896), gs_255=False, dim_ordering='default',
                 class_mode='binary', validation_mode=False,
                 balance_classes=False, all_neg_skip=0.,
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg', verbose=True):
        '''DM image iterator
        Args:
            target_size ([tuple of int]): (height, width).
            balance_classes ([bool or float]): Control class balance. When False 
                    or .0, no balancing is performed. When a float, it gives the 
                    ratio of negatives vs. positives. E.g., when balance_classes=2.0,
                    the image iterator will generate two times more negatives than 
                    positives. 
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
        self.validation_mode = validation_mode
        if validation_mode:
            balance_classes = False
            all_neg_skip = 0.
            shuffle = False
        self.balance_classes = balance_classes
        self.all_neg_skip = all_neg_skip
        self.seed = seed
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.verbose = verbose
        # Convert flattened image list.
        self.nb_sample = len(img_list)
        self.nb_class = 2
        self.filenames = list(img_list)
        self.classes = np.array(lab_list)
        nb_pos = np.sum(self.classes == 1)
        nb_neg = np.sum(self.classes == 0)
        if verbose:
            print('There are %d cancer cases and %d normal cases.' % (nb_pos, nb_neg))

        super(DMImgListIterator, self).__init__(
            self.nb_sample, batch_size, shuffle, seed)


    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
            classes_ = self.classes[index_array]
            # Obtain the random state for the current batch.
            rng = RandomState() if self.seed is None else \
                RandomState(int(self.seed) + self.total_batches_seen)
            while self.all_neg_skip > rng.uniform() and np.all(classes_ == 0):
                index_array, current_index, current_batch_size = next(self.index_generator)
                classes_ = self.classes[index_array]
                rng = RandomState() if self.seed is None else \
                    RandomState(int(self.seed) + self.total_batches_seen)

        if self.balance_classes:
            ratio = float(self.balance_classes)  # neg vs. pos.
            index_array = index_balancer(index_array, classes_, ratio, rng)

        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype='float32')

        # build batch of image data, read images first.
        last_fname = None
        for bi, ii in enumerate(index_array):  # bi: batch idx; ii: img idx.
            fname = self.filenames[ii]
            if fname == last_fname:
                batch_x[bi] = batch_x[bi-1]  # avoid repeated readings.
            else:
                last_fname = fname
                img = read_resize_img(fname, self.target_size, self.gs_255)
                # Always have one channel.
                if self.dim_ordering == 'th':
                    x = img.reshape((1, img.shape[0], img.shape[1]))
                else:
                    x = img.reshape((img.shape[0], img.shape[1], 1))
                batch_x[bi] = x

        # transform and standardize.
        for i, x in enumerate(batch_x):
            if not self.validation_mode:
                x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in xrange(current_batch_size):
                fname = '{prefix}_{index}_{hash}.{format}'.\
                    format(prefix=self.save_prefix, index=current_index + i,
                           hash=rng.randint(1e4), format=self.save_format)
                img = batch_x[i]
                if self.dim_ordering == 'th':
                    img = img.reshape((img.shape[1], img.shape[2]))
                else:
                    img = img.reshape((img.shape[0], img.shape[1]))
                # it seems only 8-bit images are supported.
                cv2.imwrite(path.join(self.save_to_dir, fname), img)
        
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype('float32')
        elif self.class_mode == 'categorical':
            batch_y = to_categorical(self.classes[index_array], self.nb_class)
        else:
            return batch_x
        return batch_x, batch_y


class DMExamListIterator(Iterator):
    '''An iterator for a flatten exam list
    '''

    def __init__(self, exam_list, image_data_generator,
                 target_size=(1152, 896), gs_255=False, dim_ordering='default',
                 class_mode='binary', validation_mode=False, prediction_mode=False, 
                 balance_classes=False, all_neg_skip=0.,
                 batch_size=16, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg', verbose=True):

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
        self.validation_mode = validation_mode
        self.prediction_mode = prediction_mode
        if validation_mode or prediction_mode:
            shuffle = False
            balance_classes = False
            all_neg_skip = 0.
        self.balance_classes = balance_classes
        self.all_neg_skip = all_neg_skip
        self.seed = seed
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.verbose = verbose
        # Convert exam list.
        self.exam_list = exam_list
        self.nb_exam = len(exam_list)
        self.nb_class = 2
        self.err_counter = 0
        # For each exam: 0 => subj id, 1 => exam idx, 2 => exam dat.
        self.classes = [ (e[2]['L']['cancer'], e[2]['R']['cancer']) 
                         for e in exam_list ]
        self.classes = np.array(self.classes)  # (exams, breasts)
        if verbose:
            print('For left breasts, normal=%d, cancer=%d, unimaged/masked=%d.' % 
                (np.sum(self.classes[:, 0] == 0), 
                 np.sum(self.classes[:, 0] == 1), 
                 np.sum(np.isnan(self.classes[:, 0])))
                )
            print('For right breasts, normal=%d, cancer=%d, unimaged/masked=%d.' % 
                (np.sum(self.classes[:, 1] == 0), 
                 np.sum(self.classes[:, 1] == 1), 
                 np.sum(np.isnan(self.classes[:, 1])))
                )

        super(DMExamListIterator, self).__init__(
            self.nb_exam, batch_size, shuffle, seed)


    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
            classes_ = np.array([ 1 if p[0] or p[1] else 0 for 
                                  p in self.classes[index_array, :] ])
            # Obtain the random state for the current batch.
            rng = RandomState() if self.seed is None else \
                RandomState(int(self.seed) + self.total_batches_seen)
            while self.all_neg_skip > rng.uniform() and np.all(classes_ == 0):
                index_array, current_index, current_batch_size = next(self.index_generator)
                classes_ = np.array([ 1 if p[0] or p[1] else 0 for 
                                      p in self.classes[index_array, :] ])                
                rng = RandomState() if self.seed is None else \
                    RandomState(int(self.seed) + self.total_batches_seen)
        if self.balance_classes:
            ratio = float(self.balance_classes)  # neg vs. pos.
            index_array = index_balancer(index_array, classes_, ratio, rng)

        # batch size measures the number of exams, an exam has two breasts.
        current_batch_size = current_batch_size*2

        # The transformation of images is not under thread lock so it can be done in parallel
        if self.prediction_mode:
            batch_x_cc = []  # a list (of breasts) of lists of image arrays.
            batch_x_mlo = []
            batch_subj = []
            batch_exam = []
        else:
            batch_x_cc = np.zeros( (current_batch_size,) + self.image_shape, dtype='float32' )
            batch_x_mlo = np.zeros( (current_batch_size,) + self.image_shape, dtype='float32' )

        def draw_img(img_df, exam=None):
            '''Read image(s) based on different modes
            Returns: a single image array or a list of image arrays
            '''
            try:
                if self.prediction_mode:
                    img = []
                    for fname in img_df['filename']:
                        img.append(read_resize_img(fname, self.target_size, self.gs_255))
                    if len(img) == 0:
                        raise ValueError('empty image dataframe')
                else:
                    if self.validation_mode:
                        fname = img_df['filename'].iloc[0]  # read the canonical view.
                    else:  # training mode.
                        fname = img_df['filename'].sample(1, random_state=rng).iloc[0]
                    img = read_resize_img(fname, self.target_size, self.gs_255)
            except ValueError:
                if self.err_counter < 10:
                    print "Error encountered reading an image dataframe:", 
                    print img_df, "Use a blank image instead."
                    print "Exam caused trouble:", exam
                    self.err_counter += 1
                img = np.zeros(self.target_size, dtype='float32')

            return img

        def read_breast_imgs(breast_dat, **kwargs):
            '''Read the images for both views for a breast
            '''
            #!!! if a view is missing, use a zero-filled 2D array.
            #!!! this may need to be changed depending on the deep learning design.
            if breast_dat['CC'] is None:
                img_cc = np.zeros(self.target_size, dtype='float32')
            else:
                img_cc = draw_img(breast_dat['CC'], **kwargs)
            if breast_dat['MLO'] is None:
                img_mlo = np.zeros(self.target_size, dtype='float32')
            else:
                img_mlo = draw_img(breast_dat['MLO'], **kwargs)
            # Convert all to lists of image arrays.
            if not isinstance(img_cc, list):
                img_cc = [img_cc]
            if not isinstance(img_mlo, list):
                img_mlo = [img_mlo]
            # Reshape each image array in the image lists.
            for i, img_cc_ in enumerate(img_cc):
                # Always have one channel.
                if self.dim_ordering == 'th':
                    img_cc[i] = img_cc_.reshape((1, img_cc_.shape[0], img_cc_.shape[1]))
                else:
                    img_cc[i] = img_cc_.reshape((img_cc_.shape[0], img_cc_.shape[1], 1))
            for i, img_mlo_ in enumerate(img_mlo):
                if self.dim_ordering == 'th':
                    img_mlo[i] = img_mlo_.reshape((1, img_mlo_.shape[0], img_mlo_.shape[1]))
                else:
                    img_mlo[i] = img_mlo_.reshape((img_mlo_.shape[0], img_mlo_.shape[1], 1))
            # Only predictin mode needs lists of image arrays.
            if not self.prediction_mode:
                img_cc = img_cc[0]
                img_mlo = img_mlo[0]

            return (img_cc, img_mlo)
            
        # build batch of image data
        adv = 0
        # last_eidx = None
        for eidx in index_array:
            # last_eidx = eidx  # no copying because sampling a diff img is expected.
            subj_id = self.exam_list[eidx][0]
            exam_idx = self.exam_list[eidx][1]
            exam_dat = self.exam_list[eidx][2]

            img_cc, img_mlo = read_breast_imgs(exam_dat['L'], exam=self.exam_list[eidx])
            if not self.prediction_mode:
                batch_x_cc[adv] = img_cc
                batch_x_mlo[adv] = img_mlo
                adv += 1
            else:
                # left_cc = img_cc
                # left_mlo = img_mlo
                batch_x_cc.append(img_cc)
                batch_x_mlo.append(img_mlo)

            img_cc, img_mlo = read_breast_imgs(exam_dat['R'], exam=self.exam_list[eidx])
            if not self.prediction_mode:
                batch_x_cc[adv] = img_cc
                batch_x_mlo[adv] = img_mlo
                adv += 1
            else:
                # right_cc = img_cc
                # right_mlo = img_mlo
                batch_x_cc.append(img_cc)
                batch_x_mlo.append(img_mlo)
                batch_subj.append(subj_id)
                batch_exam.append(exam_idx)

        # transform and standardize.
        for i in xrange(current_batch_size):
            if self.prediction_mode:
                for ii, img_cc_ in enumerate(batch_x_cc[i]):
                    if not np.all(img_cc_ == 0):
                        batch_x_cc[i][ii] = \
                            self.image_data_generator.standardize(img_cc_)
                for ii, img_mlo_ in enumerate(batch_x_mlo[i]):
                    if not np.all(img_mlo_ == 0):
                        batch_x_mlo[i][ii] = \
                            self.image_data_generator.standardize(img_mlo_)
            else:
                if not np.all(batch_x_cc[i] == 0):
                    if not self.validation_mode:
                        batch_x_cc[i] = self.image_data_generator.\
                            random_transform(batch_x_cc[i])
                    batch_x_cc[i] = self.image_data_generator.\
                        standardize(batch_x_cc[i])
                if not np.all(batch_x_mlo[i] == 0):
                    if not self.validation_mode:
                        batch_x_mlo[i] = self.image_data_generator.\
                            random_transform(batch_x_mlo[i])
                    batch_x_mlo[i] = self.image_data_generator.\
                        standardize(batch_x_mlo[i])

        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            def save_aug_img(img, bi, view, ii=None):
                '''Save an augmented image
                Args:
                    img (array): image array.
                    bi (int): breast index.
                    view (str): view name.
                    ii ([int]): (within breast) image index.
                '''
                if not self.prediction_mode:
                    fname_base = '{prefix}_{index}_{view}_{hash}'.\
                        format(prefix=self.save_prefix, index=bi, view=view, 
                               hash=rng.randint(1e4))
                else:
                    fname_base = '{prefix}_{bi}_{view}_{ii}_{hash}'.\
                        format(prefix=self.save_prefix, bi=bi, view=view, ii=ii,
                               hash=rng.randint(1e4))
                fname = fname_base + '.' + self.save_format
                if self.dim_ordering == 'th':
                    img = img.reshape((img.shape[1], img.shape[2]))
                else:
                    img = img.reshape((img.shape[0], img.shape[1]))
                # it seems only 8-bit images are supported.
                cv2.imwrite(path.join(self.save_to_dir, fname), img)


            for i in xrange(current_batch_size):
                if not self.prediction_mode:
                    img_cc = batch_x_cc[i]
                    img_mlo = batch_x_mlo[i]
                    save_aug_img(img_cc, current_index*2 + i, 'cc')
                    save_aug_img(img_mlo, current_index*2 + i, 'mlo')
                else:
                    for ii, img_cc_ in enumerate(batch_x_cc[i]):
                        save_aug_img(img_cc_, current_index*2 + i, 'cc', ii)
                    for ii, img_mlo_ in enumerate(batch_x_mlo[i]):
                        save_aug_img(img_mlo_, current_index*2 + i, 'mlo', ii)
        

        # build batch of labels
        flat_classes = self.classes[index_array, :].ravel()  # [L, R, L, R, ...]
        # flat_classes = flat_classes[np.logical_not(np.isnan(flat_classes))]
        flat_classes[np.isnan(flat_classes)] = 0  # fill in non-cancerous labels.
        if self.class_mode == 'sparse':
            batch_y = flat_classes
        elif self.class_mode == 'binary':
            batch_y = flat_classes.astype('float32')
        elif self.class_mode == 'categorical':
            batch_y = to_categorical(flat_classes, self.nb_class)
        else:  # class_mode is None.
            if self.prediction_mode:
                return [batch_subj, batch_exam, batch_x_cc, batch_x_mlo]
            else:
                return [batch_x_cc, batch_x_mlo]
        if self.prediction_mode:
            return [batch_subj, batch_exam, batch_x_cc, batch_x_mlo], batch_y
        else:
            return [batch_x_cc, batch_x_mlo], batch_y
        #### An illustration of what is returned in prediction mode: ####
        # let exam_blob = next(pred_datgen_exam)
        #
        # then           exam_blob[0][1][0][0]
        #                          /  |   \  \
        #                         /   |    \  \
        #                        /    |     \  \--- 1st img
        #                       img   cc    1st
        #                      tuple view  breast
        #
        # if class_mode is None, then the first index is not needed.


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
                           validation_mode=False,
                           balance_classes=False, all_neg_skip=0., 
                           batch_size=32, shuffle=True, seed=None,
                           save_to_dir=None, save_prefix='', save_format='jpeg', verbose=True):
        return DMImgListIterator(
            img_list, lab_list, self, 
            target_size=target_size, gs_255=gs_255, class_mode=class_mode,
            validation_mode=validation_mode,
            balance_classes=balance_classes, all_neg_skip=all_neg_skip,
            dim_ordering=self.dim_ordering,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format,
            verbose=verbose)


    def flow_from_exam_list(self, exam_list, 
                            target_size=(1152, 896), gs_255=False, class_mode='binary',
                            validation_mode=False, prediction_mode=False,
                            balance_classes=False, all_neg_skip=0., 
                            batch_size=16, shuffle=True, seed=None,
                            save_to_dir=None, save_prefix='', save_format='jpeg', verbose=True):
        return DMExamListIterator(
            exam_list, self, 
            target_size=target_size, gs_255=gs_255, class_mode=class_mode,
            validation_mode=validation_mode, prediction_mode=prediction_mode,
            balance_classes=balance_classes, all_neg_skip=all_neg_skip, 
            dim_ordering=self.dim_ordering,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format,
            verbose=verbose)

















