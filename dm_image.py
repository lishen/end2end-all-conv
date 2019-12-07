import numpy as np
from numpy.random import RandomState
from os import path
import os
from keras.preprocessing.image import (
    ImageDataGenerator, 
    Iterator, 
    # NumpyArrayIterator
)
from keras.utils.np_utils import to_categorical 
import keras.backend as K
import cv2
try:
    import dicom
except ImportError:
    import pydicom as dicom
from dm_preprocess import DMImagePreprocessor as prep
data_format = K.image_data_format()


def crop_img(img, bbox):
    '''Crop an image using bounding box
    '''
    x,y,w,h = bbox
    return img[y:y+h, x:x+w]


def add_img_margins(img, margin_size):
    '''Add all zero margins to an image
    '''
    enlarged_img = np.zeros((img.shape[0]+margin_size*2, 
                             img.shape[1]+margin_size*2))
    enlarged_img[margin_size:margin_size+img.shape[0], 
                 margin_size:margin_size+img.shape[1]] = img
    return enlarged_img


def to_sparse(y):
    '''Convert labels to sparse format if they are onehot encoded
    '''
    if y.ndim == 1:
        sparse_y = y
    elif y.ndim == 2:
        sparse_y = []
        for r in y:
            label = r.nonzero()[0]
            if len(label) != 1 or r[label[0]] != 1:
                raise ValueError('Expect one-hot encoding for y. '
                                 'Got sample:', r)
            sparse_y.append(label)
        sparse_y = np.concatenate(sparse_y)
    else:
        raise ValueError('Labels should use either sparse '
                         'or onehot encoding format. Found '
                         'shape to be: %s' % (y.shape))
    return sparse_y


def index_balancer(index_array, classes, ratio, rng):
    '''Balance an index array according to desired neg:pos ratio
    '''
    current_batch_size = len(index_array)
    pos_weight = len(classes) / (np.sum(classes==1) + 1e-7)
    neg_weight = len(classes) / (np.sum(classes!=1) + 1e-7)
    neg_weight *= ratio
    probs = np.zeros(current_batch_size)
    probs[classes==1] = pos_weight
    probs[classes!=1] = neg_weight
    probs /= probs.sum()
    index_array = rng.choice(index_array, current_batch_size, p=probs)
    index_array.sort()  # can avoid repeated img reading from disks.
    return index_array


def read_resize_img(fname, target_size=None, target_height=None, 
                    target_scale=None, gs_255=False, rescale_factor=None):
    '''Read an image (.png, .jpg, .dcm) and resize it to target size.
    '''
    if target_size is None and target_height is None:
        raise Exception('One of [target_size, target_height] must not be None')
    if path.splitext(fname)[1] == '.dcm':
        img = dicom.read_file(fname).pixel_array
    else:
        if gs_255:
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    if target_height is not None:
        target_width = int(float(target_height)/img.shape[0]*img.shape[1])
    else:
        target_height, target_width = target_size
    if (target_height, target_width) != img.shape:
        img = cv2.resize(
            img, dsize=(target_width, target_height), 
            interpolation=cv2.INTER_CUBIC)
    img = img.astype('float32')
    if target_scale is not None:
        img_max = img.max() if img.max() != 0 else target_scale
        img *= target_scale/img_max
    if rescale_factor is not None:
        img *= rescale_factor
    return img


def read_img_for_pred(fname, equalize_hist=False, data_format='channels_last', 
                      dup_3_channels=True,
                      transformer=None, standardizer=None, **kwargs):
    '''Read an image and prepare it for prediction through a network
    '''
    img = read_resize_img(fname, **kwargs)
    if equalize_hist:
        img = cv2.equalizeHist(img.astype('uint8'))
    nb_channel = 3 if dup_3_channels else 1        
    if data_format == 'channels_first':
        x = np.zeros((nb_channel,) + img.shape, dtype='float32')
        x[0,:,:] = img
        if dup_3_channels:
            x[1,:,:] = img
            x[2,:,:] = img
    else:
        x = np.zeros(img.shape + (nb_channel,), dtype='float32')
        x[:,:,0] = img
        if dup_3_channels:
            x[:,:,1] = img
            x[:,:,2] = img
    x = transformer(x) if transformer is not None else x
    x = standardizer(x) if standardizer is not None else x
    return x


def get_roi_patches(img, key_pts, roi_size=(256, 256)):
    '''Extract image patches according to a key points list
    '''
    def clip(v, minv, maxv):
        '''Clip a coordinate value to be within an image's bounds
        '''
        v = minv if v < minv else v
        v = maxv if v > maxv else v
        return v

    patches = np.zeros((len(key_pts),) + roi_size, dtype='float32')
    for i, kp in enumerate(key_pts):
        if isinstance(kp, np.ndarray):
            xc, yc = kp
        else:
            xc, yc = kp.pt
        x = int(xc - roi_size[1]/2)
        x = clip(x, 0, img.shape[1])
        y = int(yc - roi_size[0]/2)
        y = clip(y, 0, img.shape[0])
        roi = img[y:y+roi_size[0], x:x+roi_size[1]]
        patch = np.zeros(roi_size)
        patch[0:roi.shape[0], 0:roi.shape[1]] = roi
        patches[i] = patch

    return patches


def clust_kpts(key_pts, nb_clust, seed=12345):
    '''Cluster key points and return cluster centroids
    '''
    from sklearn.cluster import KMeans

    xy_coord = [ [kp.pt[0], kp.pt[1]] for kp in key_pts ]
    xy_coord = np.array(xy_coord)
    # K-means.
    clt = KMeans(nb_clust, init='k-means++', n_init=10, max_iter=30, 
                 random_state=seed)
    clt.fit(xy_coord)

    return clt.cluster_centers_


def sweep_img_patches(img, patch_size, stride, target_scale=None, 
                      equalize_hist=False):
    nb_row = round(float(img.shape[0] - patch_size)/stride + .49)
    nb_col = round(float(img.shape[1] - patch_size)/stride + .49)
    nb_row = int(nb_row)
    nb_col = int(nb_col)
    sweep_hei = patch_size + (nb_row - 1)*stride
    sweep_wid = patch_size + (nb_col - 1)*stride
    y_gap = int((img.shape[0] - sweep_hei)/2)
    x_gap = int((img.shape[1] - sweep_wid)/2)
    patch_list = []
    for y in xrange(y_gap, y_gap + nb_row*stride, stride):
        for x in xrange(x_gap, x_gap + nb_col*stride, stride):
            patch = img[y:y+patch_size, x:x+patch_size].copy()
            if target_scale is not None:
                patch_max = patch.max() if patch.max() != 0 else target_scale
                patch *= target_scale/patch_max
            if equalize_hist:
                patch = cv2.equalizeHist(patch.astype('uint8'))
            patch_list.append(patch.astype('float32'))
    return np.stack(patch_list), nb_row, nb_col


def get_prob_heatmap(img_list, target_height, target_scale, patch_size, stride, 
                     model, batch_size, 
                     featurewise_center=False, featurewise_mean=91.6,
                     preprocess=None, parallelized=False, 
                     equalize_hist=False):
    '''Sweep image data with a trained model to produce prob heatmaps
    Args:
        img_list (str or list of str): can be either an image file name or a 
            list of image file names.
    '''
    if img_list is None:
        return [None]
    elif isinstance(img_list, str):
        img_list = [img_list]
        is_single = True
    else:
        is_single = False

    heatmap_list = []
    for img_fn in img_list:
        img = read_resize_img(img_fn, target_height=target_height)
        img,_ = prep.segment_breast(img)
        img = add_img_margins(img, patch_size/2)
        patch_dat, nb_row, nb_col = sweep_img_patches(
            img, patch_size, stride, target_scale=target_scale, 
            equalize_hist=equalize_hist)
        # Make even patches if necessary.
        if parallelized and len(patch_dat) % 2 == 1:
            last_patch = patch_dat[-1:,:,:]
            patch_dat = np.append(patch_dat, last_patch, axis=0)
            appended = True
        else:
            appended = False
        if data_format == 'channels_first':
            patch_X = np.zeros((patch_dat.shape[0], 3, 
                                patch_dat.shape[1], 
                                patch_dat.shape[2]), 
                                dtype='float32')
            patch_X[:,0,:,:] = patch_dat
            patch_X[:,1,:,:] = patch_dat
            patch_X[:,2,:,:] = patch_dat
        else:
            patch_X = np.zeros((patch_dat.shape[0], 
                                patch_dat.shape[1], 
                                patch_dat.shape[2], 3), 
                                dtype='float32')
            patch_X[:,:,:,0] = patch_dat
            patch_X[:,:,:,1] = patch_dat
            patch_X[:,:,:,2] = patch_dat
        if featurewise_center:
            patch_X -= featurewise_mean
        elif preprocess is not None:
            patch_X = preprocess(patch_X)
        pred = model.predict(patch_X, batch_size=batch_size)
        if appended:  # remove the appended prediction.
            pred = pred[:-1]
        heatmap = pred.reshape((nb_row, nb_col, pred.shape[1]))
        heatmap_list.append(heatmap)
    if is_single:
        return heatmap_list[0]
    return heatmap_list 


class DMImgListIterator(Iterator):
    '''An iterator for a flatten image list
    '''

    def __init__(self, img_list, lab_list, image_data_generator,
                 target_size=(1152, 896), target_scale=4095, gs_255=False, 
                 data_format='default',
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
        if data_format == 'default':
            data_format = K.image_data_format()
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.target_scale = target_scale
        self.gs_255 = gs_255
        self.data_format = data_format
        # Always gray-scale.
        if self.data_format == 'channels_last':
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
                img = read_resize_img(
                    fname, self.target_size, target_scale=self.target_scale, 
                    gs_255=self.gs_255)
                # Always have one channel.
                if self.data_format == 'channels_first':
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
                if self.data_format == 'channels_first':
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
                 target_size=(1152, 896), target_scale=4095, gs_255=False, 
                 data_format='default',
                 class_mode='binary', validation_mode=False, prediction_mode=False, 
                 balance_classes=False, all_neg_skip=0.,
                 batch_size=16, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg', verbose=True):

        if data_format == 'default':
            data_format = K.image_data_format()
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.target_scale = target_scale
        self.gs_255 = gs_255
        self.data_format = data_format
        # Always gray-scale. Two inputs: CC and MLO.
        if self.data_format == 'channels_last':
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
                        img.append(read_resize_img(
                            fname, self.target_size, 
                            target_scale=self.target_scale, 
                            gs_255=self.gs_255))
                    if len(img) == 0:
                        raise ValueError('empty image dataframe')
                else:
                    if self.validation_mode:
                        fname = img_df['filename'].iloc[0]  # read the canonical view.
                    else:  # training mode.
                        fname = img_df['filename'].sample(1, random_state=rng).iloc[0]
                    img = read_resize_img(
                        fname, self.target_size, target_scale=self.target_scale, 
                        gs_255=self.gs_255)
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
                if self.data_format == 'channels_first':
                    img_cc[i] = img_cc_.reshape((1, img_cc_.shape[0], img_cc_.shape[1]))
                else:
                    img_cc[i] = img_cc_.reshape((img_cc_.shape[0], img_cc_.shape[1], 1))
            for i, img_mlo_ in enumerate(img_mlo):
                if self.data_format == 'channels_first':
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
                if self.data_format == 'channels_first':
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


class DMCandidROIIterator(Iterator):
    '''An iterator for candidate ROIs from mammograms
    '''

    def __init__(self, image_data_generator, img_list, lab_list=None,
                 target_height=1024, target_scale=4095, gs_255=False, 
                 data_format='default',
                 class_mode='categorical', validation_mode=False,
                 img_per_batch=2, roi_per_img=32, roi_size=(256, 256),
                 one_patch_mode=False,
                 low_int_threshold=.05, blob_min_area=3, 
                 blob_min_int=.5, blob_max_int=.85, blob_th_step=10,
                 tf_graph=None, roi_clf=None, clf_bs=32, cutpoint=.5,
                 amp_factor=1., return_sample_weight=True, 
                 auto_batch_balance=True,
                 all_neg_skip=0., shuffle=True, seed=None,
                 return_raw_img=False,
                 save_to_dir=None, save_prefix='', save_format='jpeg', 
                 verbose=True):
        '''DM candidate roi iterator
        '''
        if data_format == 'default':
            data_format = K.image_data_format()
        self.image_data_generator = image_data_generator
        self.target_height = target_height
        self.target_scale = target_scale
        self.gs_255 = gs_255
        self.data_format = data_format
        self.roi_per_img = roi_per_img
        self.roi_size = roi_size
        self.one_patch_mode = one_patch_mode
        if one_patch_mode:
            amp_factor = 1.
        self.tf_graph = tf_graph
        self.roi_clf = roi_clf
        self.clf_bs = clf_bs
        self.cutpoint = cutpoint
        if amp_factor < 1.:
            raise Exception('amp_factor must not be less than 1.0')
        self.amp_factor = amp_factor
        self.return_sample_weight = return_sample_weight
        self.auto_batch_balance = auto_batch_balance
        self.low_int_threshold = low_int_threshold
        # Always gray-scale.
        if self.data_format == 'channels_last':
            self.image_shape = self.roi_size + (1,)
        else:
            self.image_shape = (1,) + self.roi_size
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.validation_mode = validation_mode
        if validation_mode:
            all_neg_skip = 0.
            shuffle = False
        self.all_neg_skip = all_neg_skip
        self.seed = seed
        self.return_raw_img = return_raw_img
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.verbose = verbose
        # Convert flattened image list.
        self.nb_sample = len(img_list)
        self.nb_class = 3  # bkg, pos and neg.
        self.filenames = list(img_list)
        if lab_list is not None:
            self.classes = np.array(lab_list)
            nb_pos = np.sum(self.classes == 1)
            nb_neg = np.sum(self.classes == 0)
            if verbose:
                print('There are %d cancer cases and %d normal cases.' % \
                      (nb_pos, nb_neg))
        else:
            self.classes = None

        # Build a blob detector.
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = blob_min_area
        params.maxArea = roi_size[0]*roi_size[1]
        params.filterByCircularity = False
        params.filterByColor = False
        params.filterByConvexity = False
        params.filterByInertia = False
        # blob detection only works with "uint8" images.
        params.minThreshold = int(blob_min_int*255)
        params.maxThreshold = int(blob_max_int*255)
        params.thresholdStep = blob_th_step
        # import pdb; pdb.set_trace()
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3:
            self.blob_detector = cv2.SimpleBlobDetector(params)
        else:
            self.blob_detector = cv2.SimpleBlobDetector_create(params)

        super(DMCandidROIIterator, self).__init__(
            self.nb_sample, img_per_batch, shuffle, seed)


    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = \
                next(self.index_generator)
            # Obtain the random state for the current batch.
            rng = RandomState() if self.seed is None else \
                RandomState(int(self.seed) + self.total_batches_seen)
            if self.classes is not None:
                batch_cls = self.classes[index_array]
                while self.all_neg_skip > rng.uniform() and np.all(batch_cls == 0):
                    index_array, current_index, current_batch_size = \
                        next(self.index_generator)
                    batch_cls = self.classes[index_array]
                    rng = RandomState() if self.seed is None else \
                        RandomState(int(self.seed) + self.total_batches_seen)
            else:
                batch_cls = None

        # The transformation of images is not under thread lock so it can 
        # be done in parallel
        # Create a margin between pos and neg patches on cancer mammograms.
        if self.one_patch_mode:
            margin_creation = False
        else:
            margin_creation = self.amp_factor > 1. \
                              and batch_cls is not None \
                              and self.roi_clf is not None
        if margin_creation:
            nb_img_roi = int(self.roi_per_img*self.amp_factor)
        else:
            nb_img_roi = self.roi_per_img
        nb_tot_roi = current_batch_size*nb_img_roi
        batch_x = np.zeros((nb_tot_roi,) + self.image_shape, dtype='float32')

        # build batch of image data, read images first.
        batch_idx = 0
        for ii, fi in enumerate(index_array):  # ii: image idx; fi: fname idx.
            img = read_resize_img(
                self.filenames[fi], 
                target_height=self.target_height, 
                target_scale=self.target_scale, 
                gs_255=self.gs_255)
            # breast segmentation.
            img,_ = prep.segment_breast(
                img, low_int_threshold=self.low_int_threshold)
            # blob detection.
            key_pts = self.blob_detector.detect((img/img.max()*255).astype('uint8'))
            if int(self.verbose) > 1:
                print "%s: blob detection found %d key points." % \
                    (self.filenames[fi], len(key_pts))
            if len(key_pts) > nb_img_roi:
                # key_pts = rng.choice(key_pts, self.roi_per_img, replace=False)
                key_pts = clust_kpts(key_pts, nb_img_roi, self.seed)
            elif len(key_pts) > 3:  # 3 is arbitrary choice.
                key_pts = rng.choice(key_pts, nb_img_roi, replace=True)
            else:
                # blob detection failed, randomly draw points from the image.
                # import pdb; pdb.set_trace()
                pt_x = rng.randint(0, img.shape[1], nb_img_roi)
                pt_y = rng.randint(0, img.shape[0], nb_img_roi)
                key_pts = np.stack((pt_x, pt_y), axis=1)

            # get roi image patches.
            roi_patches = get_roi_patches(img, key_pts, self.roi_size)
            # Always have one channel.
            if self.data_format == 'channels_first':
                xs = roi_patches.reshape(
                    (roi_patches.shape[0], 1, roi_patches.shape[1], 
                     roi_patches.shape[2]))
            else:
                xs = roi_patches.reshape(
                    (roi_patches.shape[0], roi_patches.shape[1], 
                     roi_patches.shape[2], 1))

            # import pdb; pdb.set_trace()
            batch_x[batch_idx:batch_idx+nb_img_roi] = xs
            batch_idx += nb_img_roi

        if self.return_raw_img:
            batch_x_r = batch_x.copy()
        # transform and standardize.
        for i, x in enumerate(batch_x):
            if not self.validation_mode:
                x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in xrange(nb_tot_roi):
                fname = '{prefix}_{index}_{hash}.{format}'.\
                    format(prefix=self.save_prefix, 
                           index=current_index*nb_img_roi + i,
                           hash=rng.randint(1e4), format=self.save_format)
                img = batch_x[i]
                if self.data_format == 'channels_first':
                    img = img.reshape((img.shape[1], img.shape[2]))
                else:
                    img = img.reshape((img.shape[0], img.shape[1]))
                # it seems only 8-bit images are supported.
                cv2.imwrite(path.join(self.save_to_dir, fname), img)
        
        # score patches using the ROI classifier.
        if self.roi_clf is None:
            batch_s = np.ones((len(batch_x),))
        elif self.tf_graph is not None:
            with self.tf_graph.as_default():
                batch_s = self.roi_clf.predict(batch_x, 
                                               batch_size=self.clf_bs)
        else:
            batch_s = self.roi_clf.predict(batch_x, 
                                           batch_size=self.clf_bs)
        batch_s = batch_s.ravel()

        # use ROI prob scores to mask patches.
        if margin_creation:
            batch_mask = np.ones_like(batch_s, dtype='bool')
            batch_idx = 0
            for cls_ in batch_cls:
                img_w = batch_s[batch_idx:batch_idx+nb_img_roi]
                w_sorted_idx = np.argsort(img_w)
                img_m = np.ones_like(img_w, dtype='bool')  # per img mask.
                # filter out low score patches.
                img_m[img_w < self.cutpoint] = False
                # add back the max scored patch (in case no one passes cutpoint).
                if cls_ == 1:  # only for cancer cases.
                    img_m[w_sorted_idx[-1]] = True
                # filter out low ranked patches (if too many pass cutpoint).
                img_m[w_sorted_idx[:-self.roi_per_img]] = False
                # finally, add background patches.
                nb_bkg_patches = int(self.roi_per_img - img_m.sum())
                nb_bkg_patches = 0 if nb_bkg_patches < 0 else nb_bkg_patches
                img_m[w_sorted_idx[:nb_bkg_patches]] = True
                batch_mask[batch_idx:batch_idx+nb_img_roi] = img_m
                batch_idx += nb_img_roi
            batch_x = batch_x[batch_mask]
            batch_s = batch_s[batch_mask]
            if self.return_raw_img:
                batch_x_r = batch_x_r[batch_mask]
        elif self.one_patch_mode:
            wei_mat = batch_s.reshape((-1, self.roi_per_img))
            max_wei_idx = np.argmax(wei_mat, axis=1)
            max_wei_idx += np.arange(wei_mat.shape[0])*self.roi_per_img
            batch_mask = np.zeros_like(batch_s, dtype='bool')
            batch_mask[max_wei_idx] = True
            batch_x = batch_x[batch_mask]
            batch_s = batch_s[batch_mask]
            if self.return_raw_img:
                batch_x_r = batch_x_r[batch_mask]
        else:
            pass

        # Create background, positive and negative classes.
        if self.classes is not None:
            img_y = self.classes[index_array]
            if self.one_patch_mode:
                batch_y = img_y
            else:
                batch_y = np.array([ [y]*self.roi_per_img 
                                     for y in img_y ]).ravel()
                # Set low score patches to background.
                batch_y[batch_s < self.cutpoint] = 0
                # Add back the max scored patch (in case no one passes cutpoint).
                for ii,y in enumerate(img_y):
                    if y == 1:
                        img_idx = ii*self.roi_per_img
                        img_w = batch_s[img_idx:img_idx+self.roi_per_img]
                        max_w_idx = np.argmax(img_w)
                        batch_y[img_idx + max_w_idx] = 1
                # Assign negative patch labels.
                batch_y[np.logical_and(batch_y==0, 
                                       batch_s>=self.cutpoint)] = 2
            # In-batch balancing using sample weights.
            batch_w = np.ones_like(batch_y, dtype='float32')
            if self.auto_batch_balance:
                for uy in np.unique(batch_y):
                    batch_w[batch_y==uy] /= (batch_y==uy).mean()

        # build batch of labels
        if self.classes is None or self.class_mode is None:
            if self.return_sample_weight:
                if self.return_raw_img:
                    return batch_x_r, batch_w
                return batch_x, batch_w
            elif self.return_raw_img:
                return batch_x_r
            else:
                return batch_x
        else:
            if self.class_mode == 'sparse':
                batch_y = batch_y
            elif self.class_mode == 'binary':
                batch_y = batch_y.astype('float32')
            elif self.class_mode == 'categorical':
                batch_y = to_categorical(batch_y, self.nb_class)
            else:  # class_mode == None
                raise Exception  # this shall never happen.
        if self.return_sample_weight:
            if self.return_raw_img:
                return batch_x_r, batch_y, batch_w
            # import pdb; pdb.set_trace()
            return batch_x, batch_y, batch_w
        elif self.return_raw_img:
            return batch_x_r, batch_y
        else:
            return batch_x, batch_y


class DMNumpyArrayIterator(Iterator):

    def __init__(self, x, y, image_data_generator,
                 batch_size=32, auto_batch_balance=True, no_pos_skip=0.,
                 balance_classes=0., preprocess=None, shuffle=False, seed=None,
                 data_format='default',
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        if y is not None and len(x) != len(y):
            raise ValueError('X (images tensor) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' %
                             (np.asarray(x).shape, np.asarray(y).shape))
        if data_format == 'default':
            data_format = K.image_data_format()
        self.x = np.asarray(x, dtype=K.floatx())
        if self.x.ndim != 4:
            raise ValueError('Input data in `DMNumpyArrayIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', self.x.shape)
        channels_axis = 3 if data_format == 'channels_last' else 1
        if self.x.shape[channels_axis] not in {1, 3, 4}:
            raise ValueError('DMNumpyArrayIterator is set to use the '
                             'dimension ordering convention "' + data_format + '" '
                             '(channels on axis ' + str(channels_axis) + '), i.e. expected '
                             'either 1, 3 or 4 channels on axis ' + str(channels_axis) + '. '
                             'However, it was passed an array with shape ' + str(self.x.shape) +
                             ' (' + str(self.x.shape[channels_axis]) + ' channels).')
        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None
        self.nb_sample = len(x)
        self.image_data_generator = image_data_generator
        self.auto_batch_balance = auto_batch_balance
        self.no_pos_skip = no_pos_skip
        self.balance_classes = balance_classes
        self.preprocess = preprocess if preprocess is not None else lambda x: x
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.seed = seed
        super(DMNumpyArrayIterator, self).__init__(x.shape[0], batch_size, shuffle, seed)

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
            # Obtain the random state for the current batch.
            rng = RandomState() if self.seed is None else \
                RandomState(int(self.seed) + self.total_batches_seen)
            if self.y is not None:
                batch_y = self.y[index_array]
                sparse_y = to_sparse(batch_y)
                # import pdb; pdb.set_trace()
                while self.no_pos_skip > rng.uniform() and np.all(sparse_y != 1):
                    index_array, current_index, current_batch_size = \
                        next(self.index_generator)
                    batch_y = self.y[index_array]
                    sparse_y = to_sparse(batch_y)
                    rng = RandomState() if self.seed is None else \
                        RandomState(int(self.seed) + self.total_batches_seen)
            else:
                batch_y = None
        
        # Balance classes to over-sample pos class.
        if self.balance_classes and self.y is not None:
            ratio = float(self.balance_classes)  # neg vs. pos.
            index_array = index_balancer(index_array, sparse_y, ratio, rng)
            batch_y = self.y[index_array]
            sparse_y = to_sparse(batch_y)
            
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros(tuple([current_batch_size] + list(self.x.shape)[1:]), dtype=K.floatx())
        for i, j in enumerate(index_array):
            x = self.x[j]
            x = self.image_data_generator.random_transform(x.astype(K.floatx()))
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(path.join(self.save_to_dir, fname))
        if self.y is None:
            return self.preprocess(batch_x)
        if self.auto_batch_balance:
            batch_w = np.ones_like(sparse_y, dtype='float32')
            for uy in np.unique(sparse_y):
                batch_w[sparse_y==uy] /= (sparse_y==uy).mean()
            # import pdb; pdb.set_trace()
            return self.preprocess(batch_x), batch_y, batch_w
        return self.preprocess(batch_x), batch_y


class DMDirectoryIterator(Iterator):

    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), target_scale=None, gs_255=False,
                 equalize_hist=False, rescale_factor=None,
                 dup_3_channels=False, data_format='default',
                 classes=None, class_mode='categorical', 
                 auto_batch_balance=False, batch_size=32, 
                 preprocess=None, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg',
                 follow_links=False):
        '''
        Args:
            dup_3_channels: boolean, whether duplicate the input image onto 3 
                channels or not. This can be useful when using pretrained models 
                from databases such as ImageNet.
        '''
        if data_format == 'default':
            data_format = K.image_data_format()
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.target_scale = target_scale
        self.gs_255 = gs_255
        self.equalize_hist = equalize_hist
        self.rescale_factor = rescale_factor
        # self.xtype = 'uint8' if equalize_hist else 'float32'
        self.dup_3_channels = dup_3_channels
        self.data_format = data_format
        if self.dup_3_channels:
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
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.auto_batch_balance = auto_batch_balance
        self.preprocess = preprocess if preprocess is not None else lambda x: x
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

        # first, count the number of samples and classes
        self.nb_sample = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if path.isdir(path.join(directory, subdir)):
                    classes.append(subdir)
        self.nb_class = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        def _recursive_list(subpath):
            return sorted(os.walk(subpath, followlinks=follow_links), key=lambda tpl: tpl[0])

        for subdir in classes:
            subpath = path.join(directory, subdir)
            for root, dirs, files in _recursive_list(subpath):
                for fname in files:
                    is_valid = False
                    for extension in white_list_formats:
                        if fname.lower().endswith('.' + extension):
                            is_valid = True
                            break
                    if is_valid:
                        self.nb_sample += 1
        print('Found %d images belonging to %d classes.' % (self.nb_sample, self.nb_class))

        # second, build an index of the images in the different class subfolders
        self.filenames = []
        self.labels = np.zeros((self.nb_sample,), dtype='int32')
        i = 0
        for subdir in classes:
            subpath = path.join(directory, subdir)
            for root, dirs, files in _recursive_list(subpath):
                for fname in files:
                    is_valid = False
                    for extension in white_list_formats:
                        if fname.lower().endswith('.' + extension):
                            is_valid = True
                            break
                    if is_valid:
                        self.labels[i] = self.class_indices[subdir]
                        i += 1
                        # add filename relative to directory
                        absolute_path = path.join(root, fname)
                        self.filenames.append(path.relpath(absolute_path, directory))
        super(DMDirectoryIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape, 
                           dtype='float32')
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            x = read_img_for_pred(
                path.join(self.directory, fname), 
                equalize_hist=self.equalize_hist, data_format=self.data_format,
                dup_3_channels=self.dup_3_channels, 
                transformer=self.image_data_generator.random_transform,
                standardizer=self.image_data_generator.standardize,
                target_size=self.target_size, target_scale=self.target_scale,
                gs_255=self.gs_255, rescale_factor=self.rescale_factor)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.labels[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.labels[index_array].astype('float32')
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.nb_class), dtype='float32')
            for i, label in enumerate(self.labels[index_array]):
                batch_y[i, label] = 1.
        else:
            return self.preprocess(batch_x)
        sparse_y = self.labels[index_array]
        if self.auto_batch_balance:
            batch_w = np.ones_like(sparse_y, dtype='float32')
            for uy in np.unique(sparse_y):
                batch_w[sparse_y==uy] /= (sparse_y==uy).mean()
            return self.preprocess(batch_x), batch_y, batch_w
        return self.preprocess(batch_x), batch_y


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
                 data_format='default'):

        if data_format == 'default':
            data_format = K.image_data_format()
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
            data_format=data_format)


    def flow_from_img_list(self, img_list, lab_list, 
                           target_size=(1152, 896), target_scale=4095, gs_255=False, 
                           class_mode='binary', validation_mode=False,
                           balance_classes=False, all_neg_skip=0., 
                           batch_size=32, shuffle=True, seed=None,
                           save_to_dir=None, save_prefix='', save_format='jpeg', verbose=True):
        return DMImgListIterator(
            img_list, lab_list, self, 
            target_size=target_size, target_scale=target_scale, gs_255=gs_255, 
            class_mode=class_mode, validation_mode=validation_mode,
            balance_classes=balance_classes, all_neg_skip=all_neg_skip,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format,
            verbose=verbose)


    def flow_from_exam_list(self, exam_list, 
                            target_size=(1152, 896), target_scale=4095, gs_255=False, 
                            class_mode='binary',
                            validation_mode=False, prediction_mode=False,
                            balance_classes=False, all_neg_skip=0., 
                            batch_size=16, shuffle=True, seed=None,
                            save_to_dir=None, save_prefix='', save_format='jpeg', verbose=True):
        return DMExamListIterator(
            exam_list, self, 
            target_size=target_size, target_scale=target_scale, gs_255=gs_255, 
            class_mode=class_mode,
            validation_mode=validation_mode, prediction_mode=prediction_mode,
            balance_classes=balance_classes, all_neg_skip=all_neg_skip, 
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format,
            verbose=verbose)


    def flow_from_candid_roi(self, img_list, lab_list=None,
                 target_height=1024, target_scale=4095, gs_255=False, 
                 data_format='default',
                 class_mode='categorical', validation_mode=False,
                 img_per_batch=2, roi_per_img=32, roi_size=(256, 256),
                 one_patch_mode=False,
                 low_int_threshold=.05, blob_min_area=3, 
                 blob_min_int=.5, blob_max_int=.85, blob_th_step=10,
                 tf_graph=None, roi_clf=None, clf_bs=32, cutpoint=.5,
                 amp_factor=1., return_sample_weight=True,
                 auto_batch_balance=True,
                 all_neg_skip=0., shuffle=True, seed=None,
                 return_raw_img=False,
                 save_to_dir=None, save_prefix='', save_format='jpeg', 
                 verbose=True):
        return DMCandidROIIterator(
            self, img_list, lab_list, 
            target_height=target_height, target_scale=target_scale, 
            gs_255=gs_255, data_format=data_format,
            class_mode=class_mode, validation_mode=validation_mode,
            img_per_batch=img_per_batch, roi_per_img=roi_per_img, 
            roi_size=roi_size, one_patch_mode=one_patch_mode,
            low_int_threshold=low_int_threshold, blob_min_area=blob_min_area, 
            blob_min_int=blob_min_int, blob_max_int=blob_max_int, 
            blob_th_step=blob_th_step,
            tf_graph=tf_graph, roi_clf=roi_clf, clf_bs=clf_bs, cutpoint=cutpoint,
            amp_factor=amp_factor, return_sample_weight=return_sample_weight,
            auto_batch_balance=auto_batch_balance,
            all_neg_skip=all_neg_skip, shuffle=shuffle, seed=seed, 
            return_raw_img=return_raw_img,
            save_to_dir=save_to_dir, save_prefix=save_prefix, 
            save_format=save_format,
            verbose=verbose)


    def flow(self, X, y=None, batch_size=32, 
             auto_batch_balance=True, no_pos_skip=0., balance_classes=0.,
             preprocess=None, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg'):
        return DMNumpyArrayIterator(
            X, y, self,
            batch_size=batch_size,
            auto_batch_balance=auto_batch_balance,
            no_pos_skip = no_pos_skip,
            balance_classes=balance_classes,
            preprocess=preprocess,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)


    def flow_from_directory(self, directory,
                            target_size=(256, 256), target_scale=None, 
                            gs_255=False, equalize_hist=False,
                            rescale_factor=None,
                            dup_3_channels=False, data_format='default',
                            classes=None, class_mode='categorical',
                            auto_batch_balance=False, batch_size=32, 
                            preprocess=None, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='jpeg',
                            follow_links=False):
        return DMDirectoryIterator(
            directory, self,
            target_size=target_size, target_scale=target_scale, gs_255=gs_255,
            equalize_hist=equalize_hist, rescale_factor=rescale_factor,
            dup_3_channels=dup_3_channels, data_format=data_format,
            classes=classes, class_mode=class_mode,
            auto_batch_balance=auto_batch_balance, batch_size=batch_size, 
            preprocess=preprocess, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links)









