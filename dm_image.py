import numpy as np
from numpy.random import RandomState
from os import path
from keras.preprocessing.image import ImageDataGenerator, Iterator
from keras.utils.np_utils import to_categorical
import keras.backend as K
import cv2
import dicom
from dm_preprocess import DMImagePreprocessor as prep


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


def read_resize_img(fname, target_size=None, target_height=None, 
                    target_scale=None, gs_255=False):
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
        img *= target_scale/img.max()
    return img


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


class DMImgListIterator(Iterator):
    '''An iterator for a flatten image list
    '''

    def __init__(self, img_list, lab_list, image_data_generator,
                 target_size=(1152, 896), target_scale=4095, gs_255=False, 
                 dim_ordering='default',
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
        self.target_scale = target_scale
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
                img = read_resize_img(
                    fname, self.target_size, target_scale=self.target_scale, 
                    gs_255=self.gs_255)
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
                 target_size=(1152, 896), target_scale=4095, gs_255=False, 
                 dim_ordering='default',
                 class_mode='binary', validation_mode=False, prediction_mode=False, 
                 balance_classes=False, all_neg_skip=0.,
                 batch_size=16, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg', verbose=True):

        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.target_scale = target_scale
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


class DMCandidROIIterator(Iterator):
    '''An iterator for candidate ROIs from mammograms
    '''

    def __init__(self, image_data_generator, img_list, lab_list=None,
                 target_height=1024, target_scale=4095, gs_255=False, 
                 dim_ordering='default',
                 class_mode='binary', validation_mode=False,
                 img_per_batch=2, roi_per_img=32, roi_size=(256, 256),
                 low_int_threshold=.05, blob_min_area=3, 
                 blob_min_int=.5, blob_max_int=.85, blob_th_step=10,
                 tf_graph=None, roi_clf=None, clf_bs=32, cutpoint=.5,
                 pos_amp_factor=1., return_sample_weight=True, 
                 pos_cls_weight=1.0,
                 all_neg_skip=0., shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg', 
                 verbose=True):
        '''DM candidate roi iterator
        '''
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.image_data_generator = image_data_generator
        self.target_height = target_height
        self.target_scale = target_scale
        self.gs_255 = gs_255
        self.dim_ordering = dim_ordering
        self.roi_per_img = roi_per_img
        self.roi_size = roi_size
        self.tf_graph = tf_graph
        self.roi_clf = roi_clf
        self.clf_bs = clf_bs
        self.cutpoint = cutpoint
        if pos_amp_factor < 1.:
            raise Exception('pos_amp_factor must not be less than 1.0')
        self.pos_amp_factor = pos_amp_factor
        self.return_sample_weight = return_sample_weight
        self.pos_cls_weight = pos_cls_weight
        self.low_int_threshold = low_int_threshold
        # Always gray-scale.
        if self.dim_ordering == 'tf':
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
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.verbose = verbose
        # Convert flattened image list.
        self.nb_sample = len(img_list)
        self.nb_class = 2
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
        margin_creation = self.pos_amp_factor > 1. \
                          and batch_cls is not None \
                          and self.roi_clf is not None
        if margin_creation:
            roi_per_cancer = int(self.roi_per_img*self.pos_amp_factor)
            nb_roi = (batch_cls==1).sum()*roi_per_cancer \
                     + (batch_cls==0).sum()*self.roi_per_img
        else:
            nb_roi = current_batch_size*self.roi_per_img
        batch_x = np.zeros((nb_roi,) + self.image_shape, dtype='float32')

        # build batch of image data, read images first.
        batch_idx = 0
        for ii, fi in enumerate(index_array):  # ii: image idx; fi: fname idx.
            img = read_resize_img(
                self.filenames[fi], 
                target_height=self.target_height, 
                target_scale=self.target_scale, 
                gs_255=self.gs_255)
            # breast segmentation.
            img = prep.segment_breast(
                img, low_int_threshold=self.low_int_threshold)
            # choose number of roi.
            if margin_creation and batch_cls[ii] == 1:
                nb_img_roi = roi_per_cancer
            else:
                nb_img_roi = self.roi_per_img
            # blob detection.
            key_pts = self.blob_detector.detect((img/img.max()*255).astype('uint8'))
            if int(self.verbose) > 1:
                print "%s: blob detection found %d key points." % \
                    (self.filenames[fi], len(key_pts))
            if len(key_pts) > nb_img_roi:
                # key_pts = rng.choice(key_pts, self.roi_per_img, replace=False)
                key_pts = clust_kpts(key_pts, nb_img_roi, self.seed)
            elif len(key_pts) > 3:
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
            if self.dim_ordering == 'th':
                xs = roi_patches.reshape(
                    (1, roi_patches.shape[0], roi_patches.shape[1], 
                     roi_patches.shape[2]))
            else:
                xs = roi_patches.reshape(
                    (roi_patches.shape[0], roi_patches.shape[1], 
                     roi_patches.shape[2], 1))

            # import pdb; pdb.set_trace()
            batch_x[batch_idx:batch_idx+nb_img_roi] = xs
            batch_idx += nb_img_roi

        # transform and standardize.
        for i, x in enumerate(batch_x):
            if not self.validation_mode:
                x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in xrange(current_batch_size*self.roi_per_img):
                fname = '{prefix}_{index}_{hash}.{format}'.\
                    format(prefix=self.save_prefix, 
                           index=current_index*self.roi_per_img + i,
                           hash=rng.randint(1e4), format=self.save_format)
                img = batch_x[i]
                if self.dim_ordering == 'th':
                    img = img.reshape((img.shape[1], img.shape[2]))
                else:
                    img = img.reshape((img.shape[0], img.shape[1]))
                # it seems only 8-bit images are supported.
                cv2.imwrite(path.join(self.save_to_dir, fname), img)
        
        # calculate sample weights using the ROI classifier.
        if self.roi_clf is None:
            batch_w = np.ones((len(batch_x),))
        else:
            with self.tf_graph.as_default():
                batch_w = self.roi_clf.predict(batch_x, batch_size=self.clf_bs)
                batch_w = batch_w.ravel()

        # use ROI prob score to mask patches.
        if margin_creation:
            batch_mask = np.ones_like(batch_w, dtype='uint8')
            batch_idx = 0
            for cls_ in batch_cls:
                if cls_ == 1:
                    nb_img_roi = roi_per_cancer
                    img_w = batch_w[batch_idx:batch_idx+nb_img_roi]
                    img_m = np.ones_like(img_w, dtype='uint8')
                    # filter out low score patches.
                    img_m[img_w < self.cutpoint] = 0
                    # add negative patches.
                    nb_neg_patches = int(self.roi_per_img - img_m.sum())
                    nb_neg_patches = 0 if nb_neg_patches < 0 else nb_neg_patches
                    img_m[np.argsort(img_w)[:nb_neg_patches]] = 1
                    batch_mask[batch_idx:batch_idx+nb_img_roi] = img_m
                else:
                    nb_img_roi = self.roi_per_img
                batch_idx += nb_img_roi
            batch_mask = batch_mask.astype('bool')
            batch_x = batch_x[batch_mask]
            batch_w = batch_w[batch_mask]

        # adjust sample weights for positive class.
        if self.classes is not None:
            img_y = self.classes[index_array]
            batch_y = np.array([ [y]*self.roi_per_img for y in img_y ]).ravel()
            # Set low score patches to negative.
            batch_y[batch_w < self.cutpoint] = 0
            batch_w[batch_y==1] *= self.pos_cls_weight

        # build batch of labels
        if self.classes is None or self.class_mode is None:
            if self.return_sample_weight:
                return batch_x, batch_w
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
            return batch_x, batch_y, batch_w
        else:
            return batch_x, batch_y


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
            dim_ordering=self.dim_ordering,
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
            dim_ordering=self.dim_ordering,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format,
            verbose=verbose)


    def flow_from_candid_roi(self, img_list, lab_list=None,
                 target_height=1024, target_scale=4095, gs_255=False, 
                 dim_ordering='default',
                 class_mode='binary', validation_mode=False,
                 img_per_batch=2, roi_per_img=32, roi_size=(256, 256),
                 low_int_threshold=.05, blob_min_area=3, 
                 blob_min_int=.5, blob_max_int=.85, blob_th_step=10,
                 tf_graph=None, roi_clf=None, clf_bs=32, cutpoint=.5,
                 pos_amp_factor=1., return_sample_weight=True,
                 pos_cls_weight=1.,
                 all_neg_skip=0., shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg', 
                 verbose=True):
        return DMCandidROIIterator(
            self, img_list, lab_list, 
            target_height=target_height, target_scale=target_scale, 
            gs_255=gs_255, dim_ordering=dim_ordering,
            class_mode=class_mode, validation_mode=validation_mode,
            img_per_batch=img_per_batch, roi_per_img=roi_per_img, 
            roi_size=roi_size,
            low_int_threshold=low_int_threshold, blob_min_area=blob_min_area, 
            blob_min_int=blob_min_int, blob_max_int=blob_max_int, 
            blob_th_step=blob_th_step,
            tf_graph=tf_graph, roi_clf=roi_clf, clf_bs=clf_bs, cutpoint=cutpoint,
            pos_amp_factor=pos_amp_factor, return_sample_weight=return_sample_weight,
            pos_cls_weight=pos_cls_weight,
            all_neg_skip=all_neg_skip, shuffle=shuffle, seed=seed, 
            save_to_dir=save_to_dir, save_prefix=save_prefix, 
            save_format=save_format,
            verbose=verbose)














