import argparse, sys, os
import pickle
import numpy as np
import pandas as pd
from keras.models import load_model
# import xgboost as xgb
from meta import DMMetaManager
# from dm_image import DMImageDataGenerator
import dm_inference as dminfer
from dm_image import get_prob_heatmap
from dm_multi_gpu import make_parallel
import warnings
import exceptions
warnings.filterwarnings('ignore', category=exceptions.UserWarning)


def run(img_folder, dl_state, clf_info_state, meta_clf_state, 
        img_extension='dcm', img_height=4096, img_scale=255., 
        equalize_hist=False, featurewise_center=False, featurewise_mean=91.6,
        net='resnet50', batch_size=64, patch_size=256, stride=64,
        exam_tsv='./metadata/exams_metadata.tsv',
        img_tsv='./metadata/images_crosswalk.tsv',
        validation_mode=False, use_mean=False,
        out_pred='./output/predictions.tsv',
        progress='./progress.txt'):
    '''Run SC2 inference based on prob heatmap
    '''
    # Read some env variables.
    random_seed = int(os.getenv('RANDOM_SEED', 12345))
    rng = np.random.RandomState(random_seed)  # an rng used across board.
    gpu_count = int(os.getenv('NUM_GPU_DEVICES', 1))

    # Setup data generator for inference.
    meta_man = DMMetaManager(
        img_tsv=img_tsv, exam_tsv=exam_tsv, img_folder=img_folder, 
        img_extension='dcm')
    last2_exgen = meta_man.last_2_exam_generator()
    last2_exam_list = list(last2_exgen)
    # if do_featurewise_norm:
    #     img_gen = DMImageDataGenerator(featurewise_center=True, 
    #                                    featurewise_std_normalization=True)
    #     img_gen.mean = featurewise_mean
    #     img_gen.std = featurewise_std
    # else:
    #     img_gen = DMImageDataGenerator(samplewise_center=True, 
    #                                    samplewise_std_normalization=True)
    # if validation_mode:
    #     class_mode = 'binary'
    # else:
    #     class_mode = None

    # # Image prediction model.
    # if enet_state is not None:
    #     model = MultiViewDLElasticNet(*enet_state)
    # elif dl_state is not None:
    #     model = load_model(dl_state)
    # else:
    #     raise Exception('At least one image model state must be specified.')

    # # XGB model.
    # xgb_clf = pickle.load(open(xgb_state))

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
    meta_model = pickle.load(open(meta_clf_state))

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

    # Loop through all last 2 exam pairs.
    for i, (subj_id, curr_idx, curr_dat, prior_idx, prior_dat) in \
            enumerate(last2_exam_list):
        # DEBUG
        #if i < 23:
        #    continue
        # DEBUG
        # Get meta info for both breasts.
        left_record, right_record = meta_man.get_info_exam_pair(
            curr_dat, prior_dat)
        nb_days = left_record['daysSincePreviousExam']

        # Get image data and make predictions.
        current_exam = meta_man.get_info_per_exam(curr_dat, cc_mlo_only=True)
        if prior_idx is not None:
            prior_exam = meta_man.get_info_per_exam(prior_dat, cc_mlo_only=True)

        if validation_mode:
            left_cancer = current_exam['L']['cancer']
            right_cancer = current_exam['R']['cancer']
            left_cancer = 0 if np.isnan(left_cancer) else left_cancer
            right_cancer = 0 if np.isnan(right_cancer) else right_cancer

        # datgen_exam = img_gen.flow_from_exam_list(
        #     exam_list, target_size=(img_size[0], img_size[1]), 
        #     class_mode=class_mode, prediction_mode=True, 
        #     batch_size=len(exam_list), verbose=False)
        # ebat = next(datgen_exam)
        # if class_mode is not None:
        #     bat_x = ebat[0]
        #     bat_y = ebat[1]
        # else:
        #     bat_x = ebat
        # cc_batch = bat_x[2]
        # mlo_batch = bat_x[3]
        # curr_left_score = dminfer.pred_2view_img_list(
        #     cc_batch[0], mlo_batch[0], model, use_mean)
        # curr_right_score = dminfer.pred_2view_img_list(
        #     cc_batch[1], mlo_batch[1], model, use_mean)
        left_cc_phms = get_prob_heatmap(
            current_exam['L']['CC'], img_height, img_scale, patch_size, stride, 
            dl_model, batch_size, featurewise_center=featurewise_center, 
            featurewise_mean=featurewise_mean, preprocess=preprocess_input, 
            parallelized=parallelized, equalize_hist=equalize_hist)
        left_mlo_phms = get_prob_heatmap(
            current_exam['L']['MLO'], img_height, img_scale, patch_size, stride, 
            dl_model, batch_size, featurewise_center=featurewise_center, 
            featurewise_mean=featurewise_mean, preprocess=preprocess_input, 
            parallelized=parallelized, equalize_hist=equalize_hist)
        right_cc_phms = get_prob_heatmap(
            current_exam['R']['CC'], img_height, img_scale, patch_size, stride, 
            dl_model, batch_size, featurewise_center=featurewise_center, 
            featurewise_mean=featurewise_mean, preprocess=preprocess_input, 
            parallelized=parallelized, equalize_hist=equalize_hist)
        right_mlo_phms = get_prob_heatmap(
            current_exam['R']['MLO'], img_height, img_scale, patch_size, stride, 
            dl_model, batch_size, featurewise_center=featurewise_center, 
            featurewise_mean=featurewise_mean, preprocess=preprocess_input, 
            parallelized=parallelized, equalize_hist=equalize_hist)
        #import pdb; pdb.set_trace()
        try:
            curr_left_pred = dminfer.make_pred_case(
                left_cc_phms, left_mlo_phms, feature_name, cutoff_list, clf_list,
                k=k, nb_phm=nb_phm, use_mean=use_mean)
        except:
            curr_left_pred = 0.
        try:
            curr_right_pred = dminfer.make_pred_case(
                right_cc_phms, right_mlo_phms, feature_name, cutoff_list, clf_list,
                k=k, nb_phm=nb_phm, use_mean=use_mean)
        except:
            curr_right_pred = 0.

        if prior_idx is not None:
            # prior_left_score = dminfer.pred_2view_img_list(
            #     cc_batch[2], mlo_batch[2], model, use_mean)
            # prior_right_score = dminfer.pred_2view_img_list(
            #     cc_batch[3], mlo_batch[3], model, use_mean)
            left_cc_phms = get_prob_heatmap(
                prior_exam['L']['CC'], img_height, img_scale, patch_size, stride, 
                dl_model, batch_size, featurewise_center=featurewise_center, 
                featurewise_mean=featurewise_mean, preprocess=preprocess_input, 
                parallelized=parallelized, equalize_hist=equalize_hist)
            left_mlo_phms = get_prob_heatmap(
                prior_exam['L']['MLO'], img_height, img_scale, patch_size, stride, 
                dl_model, batch_size, featurewise_center=featurewise_center, 
                featurewise_mean=featurewise_mean, preprocess=preprocess_input, 
                parallelized=parallelized, equalize_hist=equalize_hist)
            right_cc_phms = get_prob_heatmap(
                prior_exam['R']['CC'], img_height, img_scale, patch_size, stride, 
                dl_model, batch_size, featurewise_center=featurewise_center, 
                featurewise_mean=featurewise_mean, preprocess=preprocess_input, 
                parallelized=parallelized, equalize_hist=equalize_hist)
            right_mlo_phms = get_prob_heatmap(
                prior_exam['R']['MLO'], img_height, img_scale, patch_size, stride, 
                dl_model, batch_size, featurewise_center=featurewise_center, 
                featurewise_mean=featurewise_mean, preprocess=preprocess_input, 
                parallelized=parallelized, equalize_hist=equalize_hist)
            try:
                prior_left_pred = dminfer.make_pred_case(
                    left_cc_phms, left_mlo_phms, feature_name, cutoff_list, clf_list,
                    k=k, nb_phm=nb_phm, use_mean=use_mean)
            except:
                prior_left_pred = 0.
            try:
                prior_right_pred = dminfer.make_pred_case(
                    right_cc_phms, right_mlo_phms, feature_name, cutoff_list, clf_list,
                    k=k, nb_phm=nb_phm, use_mean=use_mean)
            except:
                prior_right_pred = 0.
            diff_left_pred = (curr_left_pred - prior_left_pred)/nb_days*365
            diff_right_pred = (curr_right_pred - prior_right_pred)/nb_days*365
        else:
            prior_left_pred = np.nan
            prior_right_pred = np.nan
            diff_left_pred = np.nan
            diff_right_pred = np.nan

        # Merge image scores into meta info.
        left_record = left_record\
                .assign(curr_score=curr_left_pred)\
                .assign(prior_score=prior_left_pred)\
                .assign(diff_score=diff_left_pred)
        right_record = right_record\
                .assign(curr_score=curr_right_pred)\
                .assign(prior_score=prior_right_pred)\
                .assign(diff_score=diff_right_pred)
        #import pdb; pdb.set_trace()
        dsubj = pd.concat([left_record, right_record], ignore_index=True)

        # Predict using meta classifier.
        pred = meta_model.predict_proba(dsubj)[:,1]

        # Output.
        if validation_mode:
            fout.write("%s\t%s\tL\t%f\t%f\n" % \
                       (str(subj_id), str(curr_idx), pred[0], left_cancer))
            fout.write("%s\t%s\tR\t%f\t%f\n" % \
                       (str(subj_id), str(curr_idx), pred[1], right_cancer))
            fout.flush()
        else:
            fout.write("%s\tL\t%f\n" % (str(subj_id), pred[0]))
            fout.write("%s\tR\t%f\n" % (str(subj_id), pred[1]))
            fout.flush()

        print "processed %d/%d exams" % (i+1, len(last2_exam_list))
        sys.stdout.flush()
        with open(progress, 'w') as fpro:
            fpro.write("%f\n" % ( (i + 1.)/len(last2_exam_list)) )

    print "Done."
    fout.close()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="DM SC2 inference")
    parser.add_argument("img_folder", type=str)
    parser.add_argument("dl_state", type=str)
    parser.add_argument("clf_info_state", type=str)
    parser.add_argument("meta_clf_state", type=str)
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
    run(args.img_folder, args.dl_state, args.clf_info_state, 
        args.meta_clf_state, **run_opts)
