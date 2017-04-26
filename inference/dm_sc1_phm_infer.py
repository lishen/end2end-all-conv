import argparse, os, sys, pickle
import numpy as np
import pandas as pd
from keras.models import load_model
from meta import DMMetaManager
from dm_image import get_prob_heatmap
import dm_inference as dminfer
from dm_region import prob_heatmap_features
from dm_multi_gpu import make_parallel
import warnings
import exceptions
warnings.filterwarnings('ignore', category=exceptions.UserWarning)


def make_pred_case(cc_phms, mlo_phms, feature_name, cutoff_list, clf_list,
                   k=2, nb_phm=None, use_mean=False):
    fea_df_list = []
    for cutoff in cutoff_list:
        cc_ben_list = []
        cc_mal_list = []
        mlo_ben_list = []
        mlo_mal_list = []
        cc_fea_list = []
        mlo_fea_list = []
        for cc_phm in cc_phms[:nb_phm]:
            cc_fea_list.append(prob_heatmap_features(cc_phm, cutoff, k))
        for mlo_phm in mlo_phms[:nb_phm]:
            mlo_fea_list.append(prob_heatmap_features(mlo_phm, cutoff, k))
        for cc_fea in cc_fea_list:
            for mlo_fea in mlo_fea_list:
                cc_mal_list.append(cc_fea[0])
                cc_ben_list.append(cc_fea[1])
                mlo_mal_list.append(mlo_fea[0])
                mlo_ben_list.append(mlo_fea[1])
        cc_ben = pd.DataFrame.from_records(cc_ben_list)
        cc_mal = pd.DataFrame.from_records(cc_mal_list)
        mlo_ben = pd.DataFrame.from_records(mlo_ben_list)
        mlo_mal = pd.DataFrame.from_records(mlo_mal_list)
        cc_ben.columns = 'cc_ben_' + cc_ben.columns
        cc_mal.columns = 'cc_mal_' + cc_mal.columns
        mlo_ben.columns = 'mlo_ben_' + mlo_ben.columns
        mlo_mal.columns = 'mlo_mal_' + mlo_mal.columns
        fea_df = pd.concat([cc_ben, cc_mal, mlo_ben, mlo_mal], axis=1)
        fea_df_list.append(fea_df[feature_name])
    all_fea_df = pd.concat(fea_df_list, axis=1)
    # import pdb; pdb.set_trace()
    if len(clf_list) == 1:
        preds = clf_list[0].predict_proba(all_fea_df.values)[:,1]
    else:
        ens_clf = clf_list[0]
        pred_list = []
        for clf in clf_list[1:]:
            pred_list.append(clf.predict_proba(all_fea_df.values)[:,1])
        pred_mat = np.stack(pred_list, axis=1)
        preds = ens_clf.predict_proba(pred_mat)[:,1]
    if use_mean:
        return preds.mean()
    else:
        return preds.max()


def run(img_folder, dl_state, clf_info_state, img_extension='dcm', 
        img_height=4096, img_scale=255., 
        equalize_hist=False, featurewise_center=False, featurewise_mean=91.6,
        net='resnet50', batch_size=64, patch_size=256, stride=64,
        exam_tsv='./metadata/exams_metadata.tsv',
        img_tsv='./metadata/images_crosswalk.tsv',
        validation_mode=False, use_mean=False,
        out_pred='./output/predictions.tsv',
        progress='./progress.txt'):
    '''Run SC1 inference using prob heatmaps
    '''
    # Read some env variables.
    random_seed = int(os.getenv('RANDOM_SEED', 12345))
    rng = np.random.RandomState(random_seed)  # an rng used across board.
    gpu_count = int(os.getenv('NUM_GPU_DEVICES', 1))

    # Setup data generator for inference.
    meta_man = DMMetaManager(
        img_tsv=img_tsv, exam_tsv=exam_tsv, img_folder=img_folder, 
        img_extension=img_extension)
    if validation_mode:
        exam_list = meta_man.get_flatten_exam_list(cc_mlo_only=True)
        exam_labs = meta_man.exam_labs(exam_list)
        exam_labs = np.array(exam_labs)
        print "positive exams=%d, negative exams=%d" \
                % ((exam_labs==1).sum(), (exam_labs==0).sum())
        sys.stdout.flush()
    else:
        exam_list = meta_man.get_last_exam_list(cc_mlo_only=True)
        exam_labs = None

    # Load DL model and classifiers.
    print "Load patch classifier:", dl_state; sys.stdout.flush()
    dl_model = load_model(dl_state)
    if gpu_count > 1:
        print "Make the model parallel on %d GPUs" % (gpu_count)
        sys.stdout.flush()
        dl_model, _ = make_parallel(dl_model, gpu_count)
        parallelized = True
    else:
        parallelized = False
    feature_name, nb_phm, cutoff_list, k, clf_list = \
            pickle.load(open(clf_info_state))

    # Load preprocess function.
    if featurewise_center:
        preprocess_input = None
    else:
        print "Load preprocess function for net:", net
        if net == 'resnet50':
            from keras.applications.resnet50 import preprocess_input
        elif net == 'vgg16':
            from keras.applications.vgg16 import preprocess_input
        elif net == 'vgg19':
            from keras.applications.vgg19 import preprocess_input
        elif net == 'xception':
            from keras.applications.xception import preprocess_input
        elif net == 'inception':
            from keras.applications.inception_v3 import preprocess_input
        else:
            raise Exception("Pretrained model is not available: " + net)

    # Print header.
    fout = open(out_pred, 'w')
    if validation_mode:
        fout.write(dminfer.INFER_HEADER_VAL)
    else:
        fout.write(dminfer.INFER_HEADER)

    print "Start inference for exam list"
    sys.stdout.flush()
    for i,e in enumerate(exam_list):
        ### DEBUG ###
        #if i >= 3:
        #    break
        ### DEBUG ###
        subj = e[0]
        exam_idx = e[1]
        if validation_mode:
            left_cancer = e[2]['L']['cancer']
            right_cancer = e[2]['R']['cancer']
            left_cancer = 0 if np.isnan(left_cancer) else left_cancer
            right_cancer = 0 if np.isnan(right_cancer) else right_cancer
        left_cc_phms = get_prob_heatmap(
            e[2]['L']['CC'], img_height, img_scale, patch_size, stride, 
            dl_model, batch_size, featurewise_center=featurewise_center, 
            featurewise_mean=featurewise_mean, preprocess=preprocess_input, 
            parallelized=parallelized, equalize_hist=equalize_hist)
        left_mlo_phms = get_prob_heatmap(
            e[2]['L']['MLO'], img_height, img_scale, patch_size, stride, 
            dl_model, batch_size, featurewise_center=featurewise_center, 
            featurewise_mean=featurewise_mean, preprocess=preprocess_input, 
            parallelized=parallelized, equalize_hist=equalize_hist)
        right_cc_phms = get_prob_heatmap(
            e[2]['R']['CC'], img_height, img_scale, patch_size, stride, 
            dl_model, batch_size, featurewise_center=featurewise_center, 
            featurewise_mean=featurewise_mean, preprocess=preprocess_input, 
            parallelized=parallelized, equalize_hist=equalize_hist)
        right_mlo_phms = get_prob_heatmap(
            e[2]['R']['MLO'], img_height, img_scale, patch_size, stride, 
            dl_model, batch_size, featurewise_center=featurewise_center, 
            featurewise_mean=featurewise_mean, preprocess=preprocess_input, 
            parallelized=parallelized, equalize_hist=equalize_hist)
        left_pred = make_pred_case(
            left_cc_phms, left_mlo_phms, feature_name, cutoff_list, clf_list,
            k=k, nb_phm=nb_phm, use_mean=use_mean)
        right_pred = make_pred_case(
            right_cc_phms, right_mlo_phms, feature_name, cutoff_list, clf_list,
            k=k, nb_phm=nb_phm, use_mean=use_mean)
        if validation_mode:
            fout.write("%s\t%s\tL\t%f\t%f\n" % \
                       (str(subj), str(exam_idx), left_pred, left_cancer))
            fout.write("%s\t%s\tR\t%f\t%f\n" % \
                       (str(subj), str(exam_idx), right_pred, right_cancer))
            fout.flush()
        else:
            fout.write("%s\tL\t%f\n" % (str(subj), left_pred))
            fout.write("%s\tR\t%f\n" % (str(subj), right_pred))
            fout.flush()
        print "processed %d/%d exams" % (i+1, len(exam_list))
        sys.stdout.flush()
        with open(progress, 'w') as fpro:
            fpro.write("%f\n" % ( (i + 1.)/len(exam_list)) )
    print "Done."
    fout.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="DM SC1 inference")
    parser.add_argument("img_folder", type=str)
    parser.add_argument("dl_state", type=str)
    parser.add_argument("clf_info_state", type=str)
    parser.add_argument("--img-extension", "-ext", dest="img_extension", type=str, default="dcm")
    parser.add_argument("--img-height", dest="img_height", type=int, default=4096)
    parser.add_argument("--img-scale", dest="img_scale", type=float, default=255.)
    parser.add_argument("--equalize-hist", dest="equalize_hist", action="store_true")
    parser.add_argument("--no-equalize-hist", dest="equalize_hist", action="store_false")
    parser.set_defaults(equalize_hist=False)
    parser.add_argument("--featurewise-center", dest="featurewise_center", action="store_true")
    parser.add_argument("--no-featurewise-center", dest="featurewise_center", action="store_false")
    parser.set_defaults(featurewise_center=True)
    parser.add_argument("--featurewise-mean", dest="featurewise_mean", type=float, default=91.6)
    parser.add_argument("--net", dest="net", type=str, default="resnet50")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    parser.add_argument("--patch-size", dest="patch_size", type=int, default=256)
    parser.add_argument("--stride", dest="stride", type=int, default=64)
    parser.add_argument("--img-tsv", dest="img_tsv", type=str, default="./metadata/images_crosswalk.tsv")
    parser.add_argument("--exam-tsv", dest="exam_tsv", type=str)
    parser.add_argument("--no-exam-tsv", dest="exam_tsv", action="store_const", const=None)
    parser.set_defaults(exam_tsv=None)
    parser.add_argument("--validation-mode", dest="validation_mode", action="store_true")
    parser.add_argument("--no-validation-mode", dest="validation_mode", action="store_false")
    parser.set_defaults(validation_mode=False)
    parser.add_argument("--use-mean", dest="use_mean", action="store_true")
    parser.add_argument("--no-use-mean", dest="use_mean", action="store_false")
    parser.set_defaults(use_mean=False)
    parser.add_argument("--out-pred", dest="out_pred", type=str, default="./output/predictions.tsv")
    parser.add_argument("--progress", dest="progress", type=str, default="./progress.txt")

    args = parser.parse_args()
    run_opts = dict(
        img_extension=args.img_extension,
        img_height=args.img_height,
        img_scale=args.img_scale,
        equalize_hist=args.equalize_hist, 
        featurewise_center=args.featurewise_center,
        featurewise_mean=args.featurewise_mean,
        net=args.net,
        batch_size=args.batch_size, 
        patch_size=args.patch_size,
        stride=args.stride,
        img_tsv=args.img_tsv,
        exam_tsv=args.exam_tsv,
        validation_mode=args.validation_mode,
        use_mean=args.use_mean,
        out_pred=args.out_pred,
        progress=args.progress
    )
    print "\n>>> Inference options: <<<\n", run_opts, "\n"
    run(args.img_folder, args.dl_state, args.clf_info_state, **run_opts)












