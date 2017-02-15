import argparse
import os, sys
import pickle
import numpy as np
from numpy.random import RandomState
from scipy.sparse import lil_matrix
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from keras.models import load_model
from dm_enet import DLRepr
from meta import DMMetaManager
from dm_image import DMImageDataGenerator
from dm_keras_ext import DMMetrics as dmm

import warnings
import exceptions
warnings.filterwarnings('ignore', category=exceptions.UserWarning)


def sample_roi_repr(roi_generator, sample_per_batch, nb_samples, repr_model, 
                    batch_size=32, random_seed=12345, q_size=20):
    '''Sample candidate ROIs and then extract their DL representations
    '''
    samples_seen = 0
    repr_list = []
    roi_q = []  # a queue for candid ROIs before they are scored.
    while samples_seen < nb_samples:
        rng = RandomState(samples_seen + random_seed)
        X,w = roi_generator.next()
        w /= w.sum()
        ri = rng.choice(len(X), sample_per_batch, replace=False, p=w)
        roi_q.append(X[ri])
        samples_seen += len(ri)
        if len(roi_q) >= q_size:
            X_q = np.concatenate(roi_q)
            repr_list.append(repr_model.predict(X_q, batch_size=batch_size))
            roi_q = []
    if len(roi_q) > 0:
        X_q = np.concatenate(roi_q)
        repr_list.append(repr_model.predict(X_q, batch_size=batch_size))
        roi_q = []
    return np.concatenate(repr_list)


def get_exam_bow_mat(exam_list, nb_words, imgen, km_clf, **kw_args):
    '''Get the BoW count matrix for an exam list
    '''
    def get_roi_repr(case_all_imgs, target_height, target_scale, img_per_batch,
                     roi_per_img, roi_size, low_int_threshold, blob_min_area,
                     blob_min_int, blob_max_int, blob_th_step, seed, 
                     dlrepr_model):
        '''Get DL representations for all ROIs for all images of a case
        '''
        # import pdb; pdb.set_trace()
        roi_generator = imgen.flow_from_candid_roi(
            case_all_imgs,
            target_height=target_height, target_scale=target_scale,
            class_mode=None, validation_mode=True, 
            img_per_batch=img_per_batch, roi_per_img=roi_per_img, 
            roi_size=roi_size,
            low_int_threshold=low_int_threshold, blob_min_area=blob_min_area, 
            blob_min_int=blob_min_int, blob_max_int=blob_max_int, 
            blob_th_step=blob_th_step,
            roi_clf=None, seed=seed)

        return dlrepr_model.predict_generator(
            roi_generator, val_samples=roi_per_img*len(case_all_imgs))

    bow_mat = lil_matrix((len(exam_list)*2, nb_words), dtype='uint16')
    meta_list = []
    ri = 0
    for subj, exidx, exam in exam_list:
        try:
            cancerL = int(exam['L']['cancer'])
        except ValueError:
            cancerL = 0
        try:
            cancerR = int(exam['R']['cancer'])
        except ValueError:
            cancerR = 0
        meta_list.append((subj, exidx, 'L', cancerL))
        meta_list.append((subj, exidx, 'R', cancerR))
        try:
            clustL = km_clf.predict(get_roi_repr(exam['L']['img'], **kw_args))
            for ci in clustL:
                bow_mat[ri, ci] += 1
        except KeyError:  # unimaged breast.
            pass
        ri += 1
        try:
            clustR = km_clf.predict(get_roi_repr(exam['R']['img'], **kw_args))
            for ci in clustR:
                bow_mat[ri, ci] += 1
        except KeyError:  # unimaged breast.
            pass
        ri += 1
    return meta_list, bow_mat


def run(img_folder, dl_state, img_extension='dcm', 
        img_height=1024, img_scale=4095, val_size=.2,
        do_featurewise_norm=True, featurewise_mean=873.6, featurewise_std=739.3,
        img_per_batch=2, roi_per_img=32, roi_size=(256, 256), 
        low_int_threshold=.05, blob_min_area=3, 
        blob_min_int=.5, blob_max_int=.85, blob_th_step=10,
        roi_state=None, roi_clf_bs=32, 
        nb_pos_samples=38400, nb_neg_samples=153600, aug_for_neg=False,
        sample_per_pos=4, sample_per_neg=2, dl_clf_bs=32,
        nb_words=1024, km_max_iter=100, km_bs=1000, km_patience=20, km_init=10,
        exam_neg_vs_pos_ratio=None,
        exam_tsv='./metadata/exams_metadata.tsv',
        img_tsv='./metadata/images_crosswalk.tsv',
        km_state='./modelState/dlrepr_km_model.pkl',
        bow_train_out='./modelState/bow_dat_train.pkl',
        bow_test_out='./modelState/bow_dat_test.pkl'):
    '''Calculate bag of deep visual words count matrix for all breasts
    '''

    # Read some env variables.
    random_seed = int(os.getenv('RANDOM_SEED', 12345))

    # Load and split image and label lists.
    meta_man = DMMetaManager(exam_tsv=exam_tsv, 
                             img_tsv=img_tsv, 
                             img_folder=img_folder, 
                             img_extension=img_extension)
    subj_list, subj_labs = meta_man.get_subj_labs()
    subj_train, subj_test = train_test_split(
        subj_list, test_size=val_size, stratify=subj_labs, 
        random_state=random_seed)
    img_list, lab_list = meta_man.get_flatten_img_list(subj_train)
    img_list = np.array(img_list)
    lab_list = np.array(lab_list)
    img_pos = img_list[lab_list==1]
    img_neg = img_list[lab_list==0]
    print "Train set - Nb of positive images: %d, Nb of negative images: %d" % \
        (len(img_pos), len(img_neg))
    sys.stdout.flush()

    # Create image generator for ROIs for representation extraction.
    print "Create an image generator for ROIs"; sys.stdout.flush()
    if do_featurewise_norm:
        imgen = DMImageDataGenerator(
            featurewise_center=True, 
            featurewise_std_normalization=True)
        imgen.mean = featurewise_mean
        imgen.std = featurewise_std
    else:
        imgen = DMImageDataGenerator(
            samplewise_center=True, 
            samplewise_std_normalization=True)
    imgen.horizontal_flip = True
    imgen.vertical_flip = True
    imgen.rotation_range = 90
    imgen.width_shift_range = .25
    imgen.height_shift_range = .25
    imgen.zoom_range = [.8, 1.2]

    # Load ROI classifier.
    print "Load ROI classifier"; sys.stdout.flush()
    if roi_state is not None:
        roi_clf = load_model(
            roi_state, 
            custom_objects={
                'sensitivity': dmm.sensitivity, 
                'specificity': dmm.specificity
            }
        )
        graph = tf.get_default_graph()
    else:
        roi_clf = None
        graph = None

    # Create ROI generators for pos and neg images separately.
    print "Create ROI generators for pos and neg images"; sys.stdout.flush()
    pos_roi_generator = imgen.flow_from_candid_roi(
        img_pos,
        target_height=img_height, target_scale=img_scale,
        class_mode=None, validation_mode=False, 
        img_per_batch=img_per_batch, roi_per_img=roi_per_img, 
        roi_size=roi_size,
        low_int_threshold=low_int_threshold, blob_min_area=blob_min_area, 
        blob_min_int=blob_min_int, blob_max_int=blob_max_int, 
        blob_th_step=blob_th_step,
        tf_graph=graph, roi_clf=roi_clf, clf_bs=roi_clf_bs, 
        return_sample_weight=True, seed=random_seed)

    neg_roi_generator = imgen.flow_from_candid_roi(
        img_neg,
        target_height=img_height, target_scale=img_scale,
        class_mode=None, validation_mode=aug_for_neg,
        img_per_batch=img_per_batch, roi_per_img=roi_per_img, 
        roi_size=roi_size,
        low_int_threshold=low_int_threshold, blob_min_area=blob_min_area, 
        blob_min_int=blob_min_int, blob_max_int=blob_max_int, 
        blob_th_step=blob_th_step,
        tf_graph=graph, roi_clf=roi_clf, clf_bs=roi_clf_bs, 
        return_sample_weight=True, seed=random_seed)

    # Generate image patches and extract their DL representations.
    # Patches are sampled from candidate ROIs according to their prob scores.
    print "Load DL representation model"; sys.stdout.flush()
    dlrepr_model = DLRepr(dl_state, {'sensitivity': dmm.sensitivity, 
                                     'specificity': dmm.specificity})
    print "Sample ROIs from pos images"; sys.stdout.flush()
    pos_repr_dat = sample_roi_repr(
        pos_roi_generator, sample_per_pos*img_per_batch, nb_pos_samples, 
        dlrepr_model, batch_size=dl_clf_bs, random_seed=random_seed)
    print "Shape of pos image representation data:", pos_repr_dat.shape
    print "Sample ROIs from neg images"; sys.stdout.flush()
    neg_repr_dat = sample_roi_repr(
        neg_roi_generator, sample_per_neg*img_per_batch, nb_neg_samples, 
        dlrepr_model, batch_size=dl_clf_bs, random_seed=random_seed)
    print "Shape of neg image representation data:", neg_repr_dat.shape
    sys.stdout.flush()
    roi_repr_dat = np.concatenate([pos_repr_dat, neg_repr_dat])
    del [pos_repr_dat, neg_repr_dat]

    # Use K-means to create a codebook for deep visual words.
    print "Start K-means training on DL representations"
    rng = RandomState(random_seed)
    roi_repr_dat = rng.permutation(roi_repr_dat)  # shuffling for mini-batches.
    clf = MiniBatchKMeans(n_clusters=nb_words, init='k-means++', 
                          max_iter=km_max_iter, batch_size=km_bs, 
                          compute_labels=False, random_state=random_seed, 
                          tol=0.0, max_no_improvement=km_patience, 
                          init_size=None, n_init=km_init, 
                          reassignment_ratio=0.01, verbose=1)
    clf.fit(roi_repr_dat)
    print "K-means classifier trained:\n", clf
    sys.stdout.flush()

    # Do BoW counts for each breast and save the results.
    exam_train = meta_man.get_flatten_exam_list(
        subj_train, flatten_img_list=True)
    exam_test = meta_man.get_flatten_exam_list(
        subj_test, flatten_img_list=True)
    exam_labs_train = np.array(meta_man.exam_labs(exam_train))
    exam_labs_test = np.array(meta_man.exam_labs(exam_test))
    nb_pos_exams_train = (exam_labs_train==1).sum()
    nb_neg_exams_train = (exam_labs_train==0).sum()
    nb_pos_exams_test = (exam_labs_test==1).sum()
    nb_neg_exams_test = (exam_labs_test==0).sum()
    print "Train set - Nb of pos exams: %d, Nb of neg exams: %d" % \
        (nb_pos_exams_train, nb_neg_exams_train)
    print "Test set - Nb of pos exams: %d, Nb of neg exams: %d" % \
        (nb_pos_exams_test, nb_neg_exams_test)
    if exam_neg_vs_pos_ratio is not None:
        nb_neg_desired = int(nb_pos_exams_train*exam_neg_vs_pos_ratio)
        if nb_neg_desired < nb_neg_exams_train:
            print "Sample neg exams on train set to desired ratio:", \
                exam_neg_vs_pos_ratio; sys.stdout.flush()
            nb_neg_masked = nb_neg_exams_train - nb_neg_desired
            train_neg_mask_idx = rng.choice(
                np.where(exam_labs_train==0)[0], nb_neg_masked, replace=False)
            train_mask = np.ones_like(exam_labs_train, dtype='bool')
            train_mask[train_neg_mask_idx] = False
            exam_train = [ exam for i,exam in enumerate(exam_train) 
                           if train_mask[i] ]

    clf.set_params(verbose=0)
    print "Calculate BoW counts for train and test exam lists"
    sys.stdout.flush()
    bow_dat_train = get_exam_bow_mat(
        exam_train, nb_words, imgen, clf,
        target_height=img_height, target_scale=img_scale,
        img_per_batch=img_per_batch, roi_per_img=roi_per_img, 
        roi_size=roi_size,
        low_int_threshold=low_int_threshold, blob_min_area=blob_min_area, 
        blob_min_int=blob_min_int, blob_max_int=blob_max_int, 
        blob_th_step=blob_th_step, seed=random_seed, dlrepr_model=dlrepr_model)
    print "Shape of train BoW matrix:", bow_dat_train[1].shape
    sys.stdout.flush()
    bow_dat_test = get_exam_bow_mat(
        exam_test, nb_words, imgen, clf,
        target_height=img_height, target_scale=img_scale,
        img_per_batch=img_per_batch, roi_per_img=roi_per_img, 
        roi_size=roi_size,
        low_int_threshold=low_int_threshold, blob_min_area=blob_min_area, 
        blob_min_int=blob_min_int, blob_max_int=blob_max_int, 
        blob_th_step=blob_th_step, seed=random_seed, dlrepr_model=dlrepr_model)
    print "Shape of test BoW matrix:", bow_dat_test[1].shape
    sys.stdout.flush()

    # Save K-means model and BoW count data.
    pickle.dump(clf, open(km_state, 'w'))
    pickle.dump(bow_dat_train, open(bow_train_out, 'w'))
    pickle.dump(bow_dat_test, open(bow_test_out, 'w'))
    print "Done."


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="DM BoW training")
    parser.add_argument("img_folder", type=str)
    parser.add_argument("dl_state", type=str)
    parser.add_argument("--img-extension", "-ext", dest="img_extension", type=str, default="dcm")
    parser.add_argument("--img-height", "-ih", dest="img_height", type=int, default=1024)
    parser.add_argument("--img-scale", "-ic", dest="img_scale", type=int, default=4095)
    parser.add_argument("--val-size", "-vs", dest="val_size", type=float, default=.2)
    parser.add_argument("--featurewise-norm", dest="do_featurewise_norm", action="store_true")
    parser.add_argument("--no-featurewise-norm", dest="do_featurewise_norm", action="store_false")
    parser.set_defaults(do_featurewise_norm=True)
    parser.add_argument("--featurewise-mean", dest="featurewise_mean", type=float, default=873.6)
    parser.add_argument("--featurewise-std", dest="featurewise_std", type=float, default=739.3)
    parser.add_argument("--img-per-batch", "-ipb", dest="img_per_batch", type=int, default=2)
    parser.add_argument("--roi-per-img", "-rpi", dest="roi_per_img", type=int, default=32)
    parser.add_argument("--roi-size", dest="roi_size", nargs=2, type=int, default=[256, 256])
    parser.add_argument("--low-int-threshold", dest="low_int_threshold", type=float, default=.05)
    parser.add_argument("--blob-min-area", dest="blob_min_area", type=int, default=3)
    parser.add_argument("--blob-min-int", dest="blob_min_int", type=float, default=.5)
    parser.add_argument("--blob-max-int", dest="blob_max_int", type=float, default=.85)
    parser.add_argument("--blob-th-step", dest="blob_th_step", type=int, default=10)
    parser.add_argument("--roi-state", dest="roi_state", type=str, default=None)
    parser.add_argument("--no-roi-state", dest="roi_state", action="store_const", const=None)
    parser.add_argument("--roi-clf-bs", dest="roi_clf_bs", type=int, default=32)
    parser.add_argument("--nb-pos-samples", dest="nb_pos_samples", type=int, default=38400)
    parser.add_argument("--nb-neg-samples", dest="nb_neg_samples", type=int, default=153600)
    parser.add_argument("--aug-for-neg", dest="aug_for_neg", action="store_true")
    parser.add_argument("--no-aug-for-neg", dest="aug_for_neg", action="store_false")
    parser.set_defaults(aug_for_neg=False)
    parser.add_argument("--sample-per-pos", dest="sample_per_pos", type=int, default=4)
    parser.add_argument("--sample-per-neg", dest="sample_per_neg", type=int, default=2)
    parser.add_argument("--dl-clf-bs", dest="dl_clf_bs", type=int, default=32)
    parser.add_argument("--nb-words", dest="nb_words", type=int, default=1024)
    parser.add_argument("--km-max-iter", dest="km_max_iter", type=int, default=100)
    parser.add_argument("--km-bs", dest="km_bs", type=int, default=1000)
    parser.add_argument("--km-patience", dest="km_patience", type=int, default=20)
    parser.add_argument("--km-init", dest="km_init", type=int, default=10)
    parser.add_argument("--exam-neg-vs-pos-ratio", dest="exam_neg_vs_pos_ratio", type=float)
    parser.add_argument("--no-exam-neg-vs-pos-ratio", action="store_const", const=None)
    parser.set_defaults(exam_neg_vs_pos_ratio=None)
    parser.add_argument("--exam-tsv", "-et", dest="exam_tsv", type=str, 
                        default="./metadata/exams_metadata.tsv")
    parser.add_argument("--no-exam-tsv", dest="exam_tsv", action="store_const", const=None)
    parser.add_argument("--img-tsv", "-it", dest="img_tsv", type=str, 
                        default="./metadata/images_crosswalk.tsv")
    parser.add_argument("--km-state", dest="km_state", type=str, 
                        default="./modelState/dlrepr_km_model.pkl")
    parser.add_argument("--bow-train-out", dest="bow_train_out", type=str, 
                        default="./modelState/bow_dat_train.pkl")
    parser.add_argument("--bow-test-out", dest="bow_test_out", type=str, 
                        default="./modelState/bow_dat_test.pkl")

    args = parser.parse_args()
    run_opts = dict(
        img_extension=args.img_extension, 
        img_height=args.img_height,
        img_scale=args.img_scale,
        val_size=args.val_size if args.val_size < 1 else int(args.val_size), 
        do_featurewise_norm=args.do_featurewise_norm,
        featurewise_mean=args.featurewise_mean,
        featurewise_std=args.featurewise_std,
        img_per_batch=args.img_per_batch,
        roi_per_img=args.roi_per_img,
        roi_size=tuple(args.roi_size),
        low_int_threshold=args.low_int_threshold,
        blob_min_area=args.blob_min_area,
        blob_min_int=args.blob_min_int,
        blob_max_int=args.blob_max_int,
        blob_th_step=args.blob_th_step,
        roi_state=args.roi_state,
        roi_clf_bs=args.roi_clf_bs,
        nb_pos_samples=args.nb_pos_samples,
        nb_neg_samples=args.nb_neg_samples,
        aug_for_neg=args.aug_for_neg,
        sample_per_pos=args.sample_per_pos,
        sample_per_neg=args.sample_per_neg,
        dl_clf_bs=args.dl_clf_bs,
        nb_words=args.nb_words,
        km_max_iter=args.km_max_iter,
        km_bs=args.km_bs,
        km_patience=args.km_patience,
        km_init=args.km_init,
        exam_neg_vs_pos_ratio=args.exam_neg_vs_pos_ratio,
        exam_tsv=args.exam_tsv,
        img_tsv=args.img_tsv,
        km_state=args.km_state,
        bow_train_out=args.bow_train_out,
        bow_test_out=args.bow_test_out
    )
    print "\n>>> Model training options: <<<\n", run_opts, "\n"
    run(args.img_folder, args.dl_state, **run_opts)















