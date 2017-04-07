import argparse
import os, sys
import pickle
import numpy as np
from numpy.random import RandomState
# from sklearn.model_selection import train_test_split
from keras.models import load_model
from meta import DMMetaManager
from dm_image import add_img_margins, read_resize_img, sweep_img_patches
from dm_keras_ext import DMMetrics as dmm
from dm_multi_gpu import make_parallel
from dm_preprocess import DMImagePreprocessor as prep
import warnings
import exceptions
warnings.filterwarnings('ignore', category=exceptions.UserWarning)
import keras.backend as K
dim_ordering = K.image_dim_ordering()


def get_prob_heatmap(img_list, target_height, target_scale, patch_size, stride, 
                     model, batch_size, preprocess):
    '''Sweep image data with a trained model to produce prob heatmaps
    '''
    if img_list is None:
        return None

    heatmap_list = []
    for img_fn in img_list:
        img = read_resize_img(img_fn, target_height=target_height)
        img,_ = prep.segment_breast(img)
        img = add_img_margins(img, patch_size/2)
        patch_dat, nb_row, nb_col = sweep_img_patches(
            img, patch_size, stride, target_scale=target_scale)
        if dim_ordering == 'th':
            patch_X = np.zeros((patch_dat.shape[0], 3, 
                                patch_dat.shape[1], 
                                patch_dat.shape[2]), 
                                dtype='float64')
            patch_X[:,0,:,:] = patch_dat
            patch_X[:,1,:,:] = patch_dat
            patch_X[:,2,:,:] = patch_dat
        else:
            patch_X = np.zeros((patch_dat.shape[0], 
                                patch_dat.shape[1], 
                                patch_dat.shape[2], 3), 
                                dtype='float64')
            patch_X[:,:,:,0] = patch_dat
            patch_X[:,:,:,1] = patch_dat
            patch_X[:,:,:,2] = patch_dat
        # import pdb; pdb.set_trace()
        pred = model.predict(preprocess(patch_X), batch_size=batch_size)
        heatmap = pred.reshape((nb_row, nb_col, pred.shape[1]))
        heatmap_list.append(heatmap)
    return heatmap_list 
        

def run(img_folder, dl_state, img_extension='dcm', 
        img_height=1024, img_scale=255., neg_vs_pos_ratio=1., 
        net='vgg19', batch_size=128, patch_size=256, stride=8,
        exam_tsv='./metadata/exams_metadata.tsv',
        img_tsv='./metadata/images_crosswalk.tsv',
        out='./modelState/prob_heatmap.pkl'):
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
    subj_labs = np.array(subj_labs)
    print "Found %d subjests" % (len(subj_list))
    print "cancer patients=%d, normal patients=%d" \
            % ((subj_labs==1).sum(), (subj_labs==0).sum())
    # subj_train, subj_test, labs_train, labs_test = train_test_split(
    #     subj_list, subj_labs, test_size=test_size, stratify=subj_labs, 
    #     random_state=random_seed)
    if neg_vs_pos_ratio is not None:
        subj_list, subj_labs = DMMetaManager.subset_subj_list(
            subj_list, subj_labs, neg_vs_pos_ratio, random_seed)
        subj_labs = np.array(subj_labs)
        print "After subsetting, there are %d subjects" % (len(subj_list))
        print "cancer patients=%d, normal patients=%d" \
                % ((subj_labs==1).sum(), (subj_labs==0).sum())

    # Get exam lists.
    # >>>> Debug <<<< #
    # subj_list = subj_list[:2]
    # >>>> Debug <<<< #
    print "Get flattened exam list"
    exam_list = meta_man.get_flatten_exam_list(subj_list, cc_mlo_only=True)
    exam_labs = meta_man.exam_labs(exam_list)
    exam_labs = np.array(exam_labs)
    print "positive exams=%d, negative exams=%d" \
            % ((exam_labs==1).sum(), (exam_labs==0).sum())
    sys.stdout.flush()

    # Load DL model.
    print "Load patch classifier:", dl_state; sys.stdout.flush()
    dl_model = load_model(
        dl_state,
        custom_objects={
            'sensitivity': dmm.sensitivity, 
            'specificity': dmm.specificity
        }
    )

    # if gpu_count > 1:
    #     print "Make the model parallel on %d GPUs" % (gpu_count)
    #     sys.stdout.flush()
    #     dl_model = make_parallel(dl_model, gpu_count)

    # Load preprocess function.
    print "Load preprocess function for net:", net
    if net == 'resnet50':
        from keras.applications.resnet50 import ResNet50, preprocess_input
    elif net == 'vgg16':
        from keras.applications.vgg16 import VGG16, preprocess_input
    elif net == 'vgg19':
        from keras.applications.vgg19 import VGG19, preprocess_input
    elif net == 'xception':
        from keras.applications.xception import Xception, preprocess_input
    elif net == 'inception':
        from keras.applications.inception_v3 import InceptionV3, preprocess_input
    else:
        raise Exception("Pretrained model is not available: " + net)

    # Sweep the whole images and classify patches.
    print "Generate prob heatmaps for exam list"
    sys.stdout.flush()
    heatmap_dat_list = []
    for i,e in enumerate(exam_list):
        dat = (e[0], e[1], 
               {'L':{'cancer':e[2]['L']['cancer']}, 
                'R':{'cancer':e[2]['R']['cancer']}})
        dat[2]['L']['CC'] = get_prob_heatmap(
            e[2]['L']['CC'], img_height, img_scale, patch_size, stride, 
            dl_model, batch_size, preprocess_input)
        dat[2]['L']['MLO'] = get_prob_heatmap(
            e[2]['L']['MLO'], img_height, img_scale, patch_size, stride, 
            dl_model, batch_size, preprocess_input)
        dat[2]['R']['CC'] = get_prob_heatmap(
            e[2]['R']['CC'], img_height, img_scale, patch_size, stride, 
            dl_model, batch_size, preprocess_input)
        dat[2]['R']['MLO'] = get_prob_heatmap(
            e[2]['R']['MLO'], img_height, img_scale, patch_size, stride, 
            dl_model, batch_size, preprocess_input)
        heatmap_dat_list.append(dat)
        print "processed %d/%d exams" % (i+1, len(exam_list))
        sys.stdout.flush()
        ### DEBUG ###
        #if i >= 9:
        #    break
        ### DEBUG ###
    print "Done."

    # Save the result.
    print "Saving result to external files.",
    sys.stdout.flush()
    pickle.dump(heatmap_dat_list, open(out, 'w'))
    print "Done."


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="DM prob heatmap scoring")
    parser.add_argument("img_folder", type=str)
    parser.add_argument("dl_state", type=str)
    parser.add_argument("--img-extension", "-ext", dest="img_extension", type=str, default="dcm")
    parser.add_argument("--img-height", "-ih", dest="img_height", type=int, default=1024)
    parser.add_argument("--img-scale", "-ic", dest="img_scale", type=float, default=255.)
    parser.add_argument("--neg-vs-pos-ratio", dest="neg_vs_pos_ratio", type=float, default=10.)
    parser.add_argument("--no-neg-vs-pos-ratio", dest="neg_vs_pos_ratio", 
                        action="store_const", const=None)
    parser.add_argument("--net", dest="net", type=str, default="vgg19")    
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=128)
    parser.add_argument("--patch-size", dest="patch_size", type=int, default=256)
    parser.add_argument("--stride", dest="stride", type=int, default=8)
    parser.add_argument("--exam-tsv", "-et", dest="exam_tsv", type=str, 
                        default="./metadata/exams_metadata.tsv")
    parser.add_argument("--no-exam-tsv", dest="exam_tsv", action="store_const", const=None)
    parser.add_argument("--img-tsv", "-it", dest="img_tsv", type=str, 
                        default="./metadata/images_crosswalk.tsv")
    parser.add_argument("--out", dest="out", type=str, 
                        default="./modelState/prob_heatmap.pkl")

    args = parser.parse_args()
    run_opts = dict(
        img_extension=args.img_extension, 
        img_height=args.img_height,
        img_scale=args.img_scale,
        neg_vs_pos_ratio=args.neg_vs_pos_ratio,
        net=args.net,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        stride=args.stride,
        exam_tsv=args.exam_tsv,
        img_tsv=args.img_tsv,
        out=args.out,
    )
    print "\n"
    print "img_folder=%s" % (args.img_folder)
    print "dl_state=%s" % (args.dl_state)
    print "\n>>> Model training options: <<<\n", run_opts, "\n"
    run(args.img_folder, args.dl_state, **run_opts)















