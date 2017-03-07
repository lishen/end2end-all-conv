import argparse
import os, sys
import pickle
import numpy as np
from numpy.random import RandomState
from sklearn.model_selection import train_test_split
from keras.models import load_model
from meta import DMMetaManager
from dm_image import DMImageDataGenerator
from dm_keras_ext import DMMetrics as dmm
from dm_multi_gpu import make_parallel

import warnings
import exceptions
warnings.filterwarnings('ignore', category=exceptions.UserWarning)


def get_exam_pred(exam_list, roi_per_img, imgen, **kw_args):
    '''Get the predictions for an exam list
    '''
    #####################################################
    def get_breast_prob(
        case_all_imgs, target_height, target_scale, 
        img_per_batch, roi_size, 
        low_int_threshold, blob_min_area, blob_min_int, blob_max_int, 
        blob_th_step, seed, dl_model):
        '''Get prob for all ROIs for all images of a case
        '''
        roi_generator = imgen.flow_from_candid_roi(
            case_all_imgs,
            target_height=target_height, target_scale=target_scale,
            class_mode=None, validation_mode=True, 
            img_per_batch=img_per_batch, roi_per_img=roi_per_img, 
            roi_size=roi_size,
            low_int_threshold=low_int_threshold, blob_min_area=blob_min_area, 
            blob_min_int=blob_min_int, blob_max_int=blob_max_int, 
            blob_th_step=blob_th_step,
            roi_clf=None, return_sample_weight=False, seed=seed)
        # import pdb; pdb.set_trace()
        pred = dl_model.predict_generator(
            roi_generator, val_samples=roi_per_img*len(case_all_imgs))
        # New shape: img x roi x output.
        pred = pred.reshape((len(case_all_imgs), roi_per_img, -1))
        return pred
    #####################################################

    meta_prob_list = []
    for subj, exidx, exam in exam_list:
        try:
            cancerL = int(exam['L']['cancer'])
        except ValueError:
            cancerL = 0
        try:
            cancerR = int(exam['R']['cancer'])
        except ValueError:
            cancerR = 0

        try:
            probL = get_breast_prob(exam['L']['img'], **kw_args)
        except KeyError:  # unimaged breast.
            probL = np.array([[[1.,0.,0.]]*roi_per_img])
        meta_prob_list.append((subj, exidx, 'L', cancerL, probL))

        try:
            probR = get_breast_prob(exam['R']['img'], **kw_args)
        except KeyError:  # unimaged breast.
            probR = np.array([[[1.,0.,0.]]*roi_per_img])
        meta_prob_list.append((subj, exidx, 'R', cancerR, probR))

    return meta_prob_list


def run(img_folder, dl_state, img_extension='dcm', 
        img_height=1024, img_scale=4095, val_size=.2, neg_vs_pos_ratio=10., 
        do_featurewise_norm=True, featurewise_mean=873.6, featurewise_std=739.3,
        img_per_batch=2, roi_per_img=32, roi_size=(256, 256), 
        low_int_threshold=.05, blob_min_area=3, 
        blob_min_int=.5, blob_max_int=.85, blob_th_step=10,
        exam_tsv='./metadata/exams_metadata.tsv',
        img_tsv='./metadata/images_crosswalk.tsv',
        train_out='./modelState/meta_prob_train.pkl',
        test_out='./modelState/meta_prob_test.pkl'):
    '''Calculate bag of deep visual words count matrix for all breasts
    '''

    # Read some env variables.
    random_seed = int(os.getenv('RANDOM_SEED', 12345))
    rng = RandomState(random_seed)  # an rng used across board.
    gpu_count = int(os.getenv('NUM_GPU_DEVICES', 1))

    # Load and split image and label lists.
    meta_man = DMMetaManager(exam_tsv=exam_tsv, 
                             img_tsv=img_tsv, 
                             img_folder=img_folder, 
                             img_extension=img_extension)
    subj_list, subj_labs = meta_man.get_subj_labs()
    subj_train, subj_test, labs_train, labs_test = train_test_split(
        subj_list, subj_labs, test_size=val_size, stratify=subj_labs, 
        random_state=random_seed)
    if neg_vs_pos_ratio is not None:
        def subset_subj(subj, labs):
            subj = np.array(subj)
            labs = np.array(labs)
            pos_idx = np.where(labs==1)[0]
            neg_idx = np.where(labs==0)[0]
            nb_neg_desired = int(len(pos_idx)*neg_vs_pos_ratio)
            if nb_neg_desired >= len(neg_idx):
                return subj.tolist()
            else:
                neg_chosen = rng.choice(neg_idx, nb_neg_desired, replace=False)
                subset_idx = np.concatenate([pos_idx, neg_chosen])
                return subj[subset_idx].tolist()

        subj_train = subset_subj(subj_train, labs_train)
        subj_test = subset_subj(subj_test, labs_test)

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

    # Load DL model.
    print "Load DL classification model:", dl_state; sys.stdout.flush()
    dl_model = load_model(
        dl_state, 
        custom_objects={
            'sensitivity': dmm.sensitivity, 
            'specificity': dmm.specificity
        }
    )
    if gpu_count > 1:
        print "Make the model parallel on %d GPUs" % (gpu_count)
        sys.stdout.flush()
        dl_model = make_parallel(dl_model, gpu_count)

    # Read exam lists.
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

    # Make predictions for exam lists.
    print "Predicting for train exam list"; sys.stdout.flush()
    meta_prob_train = get_exam_pred(
        exam_train, roi_per_img, imgen, 
        target_height=img_height, target_scale=img_scale,
        img_per_batch=img_per_batch, roi_size=roi_size,
        low_int_threshold=low_int_threshold, blob_min_area=blob_min_area, 
        blob_min_int=blob_min_int, blob_max_int=blob_max_int, 
        blob_th_step=blob_th_step, seed=random_seed, 
        dl_model=dl_model)
    print "Length of train prediction list:", len(meta_prob_train)
    sys.stdout.flush()

    print "Predicting for test exam list"; sys.stdout.flush()
    meta_prob_test = get_exam_pred(
        exam_test, roi_per_img, imgen, 
        target_height=img_height, target_scale=img_scale,
        img_per_batch=img_per_batch, roi_size=roi_size,
        low_int_threshold=low_int_threshold, blob_min_area=blob_min_area, 
        blob_min_int=blob_min_int, blob_max_int=blob_max_int, 
        blob_th_step=blob_th_step, seed=random_seed, 
        dl_model=dl_model)
    print "Length of test prediction list:", len(meta_prob_test)
    sys.stdout.flush()

    pickle.dump(meta_prob_train, open(train_out, 'w'))
    pickle.dump(meta_prob_test, open(test_out, 'w'))
    print "Done."


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="DM candidROI prediction")
    parser.add_argument("img_folder", type=str)
    parser.add_argument("dl_state", type=str)
    parser.add_argument("--img-extension", "-ext", dest="img_extension", type=str, default="dcm")
    parser.add_argument("--img-height", "-ih", dest="img_height", type=int, default=1024)
    parser.add_argument("--img-scale", "-ic", dest="img_scale", type=int, default=4095)
    parser.add_argument("--val-size", "-vs", dest="val_size", type=float, default=.2)
    parser.add_argument("--neg-vs-pos-ratio", dest="neg_vs_pos_ratio", type=float, default=10.)
    parser.add_argument("--no-neg-vs-pos-ratio", dest="neg_vs_pos_ratio", 
                        action="store_const", const=None)
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
    parser.add_argument("--exam-tsv", "-et", dest="exam_tsv", type=str, 
                        default="./metadata/exams_metadata.tsv")
    parser.add_argument("--no-exam-tsv", dest="exam_tsv", action="store_const", const=None)
    parser.add_argument("--img-tsv", "-it", dest="img_tsv", type=str, 
                        default="./metadata/images_crosswalk.tsv")
    parser.add_argument("--train-out", dest="train_out", type=str, 
                        default="./modelState/meta_prob_train.pkl")
    parser.add_argument("--test-out", dest="test_out", type=str, 
                        default="./modelState/meta_prob_test.pkl")

    args = parser.parse_args()
    run_opts = dict(
        img_extension=args.img_extension, 
        img_height=args.img_height,
        img_scale=args.img_scale,
        val_size=args.val_size if args.val_size < 1 else int(args.val_size), 
        neg_vs_pos_ratio=args.neg_vs_pos_ratio,
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
        exam_tsv=args.exam_tsv,
        img_tsv=args.img_tsv,
        train_out=args.train_out,
        test_out=args.test_out
    )
    print "\n>>> Model training options: <<<\n", run_opts, "\n"
    run(args.img_folder, args.dl_state, **run_opts)















