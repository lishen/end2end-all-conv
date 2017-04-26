import argparse
import os, sys
import pickle
import numpy as np
import pandas as pd
from numpy.random import RandomState
from sklearn.model_selection import train_test_split
from keras.models import load_model
from dm_image import add_img_margins, read_resize_img, sweep_img_patches
from dm_keras_ext import DMMetrics as dmm, get_dl_model
from dm_multi_gpu import make_parallel
from dm_preprocess import DMImagePreprocessor as prep
import warnings
import exceptions
warnings.filterwarnings('ignore', category=exceptions.UserWarning)
import keras.backend as K
dim_ordering = K.image_dim_ordering()


def get_prob_heatmap(img_fn, target_height, target_scale, patch_size, stride, 
                     model, batch_size, preprocess):
    '''Sweep image data with a trained model to produce prob heatmaps
    '''
    img = read_resize_img(img_fn, target_height=target_height)
    img,_ = prep.segment_breast(img)
    img = add_img_margins(img, patch_size/2)
    patch_dat, nb_row, nb_col = sweep_img_patches(
        img, patch_size, stride, target_scale=target_scale)
    # import pdb; pdb.set_trace()
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
    pred = model.predict(preprocess(patch_X), batch_size=batch_size)
    heatmap = pred.reshape((nb_row, nb_col, pred.shape[1]))
    return heatmap
        

def run(img_folder, dl_state, 
        img_height=1024, img_scale=255., 
        net='vgg19', batch_size=128, patch_size=256, stride=8,
        pat_csv='./full_img/pat.csv', pat_list=None,
        out='./full_img/prob_heatmap.pkl'):
    '''Sweep mammograms with trained DL model to create prob heatmaps
    '''
    # Read some env variables.
    random_seed = int(os.getenv('RANDOM_SEED', 12345))
    rng = RandomState(random_seed)  # an rng used across board.
    #gpu_count = int(os.getenv('NUM_GPU_DEVICES', 1))

    # Get image and label lists.
    df = pd.read_csv(pat_csv, header=0)
    df = df.set_index(['patient_id', 'side'])
    df.sort_index(inplace=True)
    if pat_list is not None:
        pat_ids = pd.read_csv(pat_list, header=0).values.ravel()
        pat_ids = pat_ids.tolist()
        print "Read %d patient IDs" % (len(pat_ids))
        df = df.loc[pat_ids]

    # Load DL model, preprocess.
    print "Load patch classifier:", dl_state; sys.stdout.flush()
    dl_model, preprocess_input, _ = get_dl_model(
        net, use_pretrained=True, resume_from=dl_state)
    #if gpu_count > 1:
    #    print "Make the model parallel on %d GPUs" % (gpu_count)
    #    sys.stdout.flush()
    #    dl_model = make_parallel(dl_model, gpu_count)

    # Sweep the whole images and classify patches.
    def const_filename(pat, side, view):
        basename = '_'.join([pat, side, view]) + '.png'
        return os.path.join(img_folder, basename)
    print "Generate prob heatmaps"; sys.stdout.flush()
    heatmaps = []
    cases_seen = 0
    nb_cases = len(df.index.unique())
    for pat,side in df.index.unique():
        cancer = df.loc[pat].loc[side]['cancer']
        cc_fn = const_filename(pat, side, 'CC')
        if os.path.isfile(cc_fn):
            cc_hm = get_prob_heatmap(
                cc_fn, img_height, img_scale, patch_size, stride, 
                dl_model, batch_size, preprocess_input)
        else:
            cc_hm = None
        mlo_fn = const_filename(pat, side, 'MLO')
        if os.path.isfile(mlo_fn):
            mlo_hm = get_prob_heatmap(
                mlo_fn, img_height, img_scale, patch_size, stride, 
                dl_model, batch_size, preprocess_input)
        else:
            mlo_hm = None
        heatmaps.append({'patient_id':pat, 'side':side, 'cancer':cancer, 
                         'cc':cc_hm, 'mlo':mlo_hm})
        cases_seen += 1
        print "scored %d/%d cases" % (cases_seen, nb_cases)
        sys.stdout.flush()
    print "Done."

    # Save the result.
    print "Saving result to external files.",
    sys.stdout.flush()
    pickle.dump(heatmaps, open(out, 'w'))
    print "Done."


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="heatmap scoring")
    parser.add_argument("img_folder", type=str)
    parser.add_argument("dl_state", type=str)
    parser.add_argument("--img-height", "-ih", dest="img_height", type=int, default=1024)
    parser.add_argument("--img-scale", "-ic", dest="img_scale", type=float, default=255.)
    parser.add_argument("--net", dest="net", type=str, default="vgg19")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=128)
    parser.add_argument("--patch-size", dest="patch_size", type=int, default=256)
    parser.add_argument("--stride", dest="stride", type=int, default=8)
    parser.add_argument("--pat-csv", dest="pat_csv", type=str, default="./full_img/pat.csv")
    parser.add_argument("--pat-list", dest="pat_list", type=str, default=None)
    parser.add_argument("--no-pat-list", dest="pat_list", action="store_const", const=None)
    parser.add_argument("--out", dest="out", type=str, default="./full_img/prob_heatmap.pkl")

    args = parser.parse_args()
    run_opts = dict(
        img_height=args.img_height,
        img_scale=args.img_scale,
        net=args.net,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        stride=args.stride,
        pat_csv=args.pat_csv,
        pat_list=args.pat_list,
        out=args.out,
    )
    print "\n"
    print "img_folder=%s" % (args.img_folder)
    print "dl_state=%s" % (args.dl_state)
    print "\n>>> Model training options: <<<\n", run_opts, "\n"
    run(args.img_folder, args.dl_state, **run_opts)















