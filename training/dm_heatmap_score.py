import argparse
import os, sys
import pickle
import numpy as np
from numpy.random import RandomState
from sklearn.model_selection import train_test_split
from keras.models import load_model
from meta import DMMetaManager
from dm_image import DMImageDataGenerator, add_img_margins, read_resize_img
from dm_keras_ext import DMMetrics as dmm
from dm_multi_gpu import make_parallel
from dm_preprocess import DMImagePreprocessor as prep
import warnings
import exceptions
warnings.filterwarnings('ignore', category=exceptions.UserWarning)
import keras.backend as K
dim_ordering = K.image_dim_ordering()


def get_prob_heatmap(img_list, target_height, target_scale, patch_size, stride, 
                     model, batch_size,
                     do_featurewise_norm=True, 
                     featurewise_mean=873.6, featurewise_std=739.3):
    '''Sweep image data with a trained model to produce prob heatmaps
    '''
    def sweep_img_patches(img):
        nb_row = int(float(img.shape[0] - patch_size)/stride)
        nb_col = int(float(img.shape[1] - patch_size)/stride)
        sweep_hei = patch_size + (nb_row - 1)*stride
        sweep_wid = patch_size + (nb_col - 1)*stride
        y_gap = int((img.shape[0] - sweep_hei)/2)
        x_gap = int((img.shape[1] - sweep_wid)/2)
        patch_list = []
        for y in xrange(y_gap, y_gap + nb_row*stride, stride):
            for x in xrange(x_gap, x_gap + nb_col*stride, stride):
                patch = img[y:y+patch_size, x:x+patch_size]
                if do_featurewise_norm:
                    patch = (patch - featurewise_mean)/featurewise_std
                else:
                    patch = (patch - patch.mean())/patch.std()
                patch_list.append(patch)
        return np.stack(patch_list), nb_row, nb_col


    heatmap_list = []
    for img_fn in img_list:
        img = read_resize_img(
            img_fn, target_height=target_height, target_scale=target_scale)
        img,_ = prep.segment_breast(img)
        img = add_img_margins(img, patch_size/2)
        patch_dat, nb_row, nb_col = sweep_img_patches(img)
        if dim_ordering == 'th':
            patch_dat = patch_dat.reshape(
                (patch_dat.shape[0], 1, 
                 patch_dat.shape[1], 
                 patch_dat.shape[2]))
        else:
            patch_dat = patch_dat.reshape(
                (patch_dat.shape[0], 
                 patch_dat.shape[1], 
                 patch_dat.shape[2], 1))
        pred = model.predict(patch_dat, batch_size=batch_size)
        heatmap = pred.reshape((nb_row, nb_col, pred.shape[1]))
        heatmap_list.append(heatmap)
    return heatmap_list 
        

def run(img_folder, dl_state, img_extension='dcm', 
        img_height=1024, img_scale=4095, test_size=.2, neg_vs_pos_ratio=10., 
        do_featurewise_norm=True, featurewise_mean=873.6, featurewise_std=739.3,
        batch_size=128, patch_size=256, stride=8,
        exam_tsv='./metadata/exams_metadata.tsv',
        img_tsv='./metadata/images_crosswalk.tsv',
        train_out='./modelState/prob_heatmap_train.pkl',
        test_out='./modelState/prob_heatmap_test.pkl'):
    '''Sweep mammograms with trained DL model to create prob heatmaps
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
        subj_list, subj_labs, test_size=test_size, stratify=subj_labs, 
        random_state=random_seed)
    if neg_vs_pos_ratio is not None:
        subj_train, labs_train = DMMetaManager.subset_subj_list(
            subj_train, labs_train, neg_vs_pos_ratio, random_seed)

    # Get image and label lists.
    # >>>> Debug <<<< #
    subj_train = subj_train[:2]
    subj_test = subj_test[:2]
    # >>>> Debug <<<< #
    img_train, ilab_train = meta_man.get_flatten_img_list(subj_train)
    img_test, ilab_test = meta_man.get_flatten_img_list(subj_test)

    # Load DL model.
    print "Load patch classifier:", dl_state; sys.stdout.flush()
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

    # Sweep the whole images and classify patches.
    print "Generate prob heatmaps for train set.",
    sys.stdout.flush()
    train_heatmaps = get_prob_heatmap(
        img_train, img_height, img_scale, 
        patch_size, stride, dl_model, batch_size, 
        do_featurewise_norm, featurewise_mean, featurewise_std)
    print "Done."
    print "Generate prob heatmaps for test set.",
    sys.stdout.flush()
    test_heatmaps = get_prob_heatmap(
        img_test, img_height, img_scale, 
        patch_size, stride, dl_model, batch_size, 
        do_featurewise_norm, featurewise_mean, featurewise_std)
    print "Done."

    # Save the result.
    print "Saving result to external files.",
    sys.stdout.flush()
    pickle.dump((train_heatmaps, img_train, ilab_train), open(train_out, 'w'))
    pickle.dump((test_heatmaps, img_test, ilab_test), open(test_out, 'w'))
    print "Done."


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="DM candidROI prediction")
    parser.add_argument("img_folder", type=str)
    parser.add_argument("dl_state", type=str)
    parser.add_argument("--img-extension", "-ext", dest="img_extension", type=str, default="dcm")
    parser.add_argument("--img-height", "-ih", dest="img_height", type=int, default=1024)
    parser.add_argument("--img-scale", "-ic", dest="img_scale", type=int, default=4095)
    parser.add_argument("--test-size", "-ts", dest="test_size", type=float, default=.2)
    parser.add_argument("--neg-vs-pos-ratio", dest="neg_vs_pos_ratio", type=float, default=10.)
    parser.add_argument("--no-neg-vs-pos-ratio", dest="neg_vs_pos_ratio", 
                        action="store_const", const=None)
    parser.add_argument("--featurewise-norm", dest="do_featurewise_norm", action="store_true")
    parser.add_argument("--no-featurewise-norm", dest="do_featurewise_norm", action="store_false")
    parser.set_defaults(do_featurewise_norm=True)
    parser.add_argument("--featurewise-mean", dest="featurewise_mean", type=float, default=873.6)
    parser.add_argument("--featurewise-std", dest="featurewise_std", type=float, default=739.3)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=128)
    parser.add_argument("--patch-size", dest="patch_size", type=int, default=256)
    parser.add_argument("--stride", dest="stride", type=int, default=8)
    parser.add_argument("--exam-tsv", "-et", dest="exam_tsv", type=str, 
                        default="./metadata/exams_metadata.tsv")
    parser.add_argument("--no-exam-tsv", dest="exam_tsv", action="store_const", const=None)
    parser.add_argument("--img-tsv", "-it", dest="img_tsv", type=str, 
                        default="./metadata/images_crosswalk.tsv")
    parser.add_argument("--train-out", dest="train_out", type=str, 
                        default="./modelState/prob_heatmap_train.pkl")
    parser.add_argument("--test-out", dest="test_out", type=str, 
                        default="./modelState/prob_heatmap_test.pkl")

    args = parser.parse_args()
    run_opts = dict(
        img_extension=args.img_extension, 
        img_height=args.img_height,
        img_scale=args.img_scale,
        test_size=args.test_size if args.test_size < 1 else int(args.test_size), 
        neg_vs_pos_ratio=args.neg_vs_pos_ratio,
        do_featurewise_norm=args.do_featurewise_norm,
        featurewise_mean=args.featurewise_mean,
        featurewise_std=args.featurewise_std,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        stride=args.stride,
        exam_tsv=args.exam_tsv,
        img_tsv=args.img_tsv,
        train_out=args.train_out,
        test_out=args.test_out
    )
    print "\n"
    print "img_folder=%s" % (args.img_folder)
    print "dl_state=%s" % (args.dl_state)
    print "\n>>> Model training options: <<<\n", run_opts, "\n"
    run(args.img_folder, args.dl_state, **run_opts)















