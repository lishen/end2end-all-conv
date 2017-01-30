import argparse
import pickle
import numpy as np
import pandas as pd
from keras.models import load_model
import xgboost as xgb
from meta import DMMetaManager
from dm_image import DMImageDataGenerator
from dm_enet import MultiViewDLElasticNet
import dm_inference as dminfer

import warnings
import exceptions
warnings.filterwarnings('ignore', category=exceptions.UserWarning)


def run(img_folder, img_size=[288, 224], do_featurewise_norm=True, 
        featurewise_mean=485.9, featurewise_std=765.2, 
        img_tsv='./metadata/images_crosswalk.tsv',
        exam_tsv='./metadata/exams_metadata.tsv',
        dl_state=None,
        enet_state=None,
        xgb_state=None,
        validation_mode=False, use_mean=False,
        out_pred='./output/predictions.tsv'):
    '''Run SC2 inference
    Args:
        featurewise_mean, featurewise_std ([float]): they are estimated from 
                1152 x 896 images. Using different sized images give very close
                results. For png, mean=7772, std=12187.
    '''

    # Setup data generator for inference.
    meta_man = DMMetaManager(
        img_tsv=img_tsv, exam_tsv=exam_tsv, img_folder=img_folder, 
        img_extension='dcm')
    last2_exgen = meta_man.last_2_exam_generator()
    if do_featurewise_norm:
        img_gen = DMImageDataGenerator(featurewise_center=True, 
                                       featurewise_std_normalization=True)
        img_gen.mean = featurewise_mean
        img_gen.std = featurewise_std
    else:
        img_gen = DMImageDataGenerator(samplewise_center=True, 
                                       samplewise_std_normalization=True)
    if validation_mode:
        class_mode = 'binary'
    else:
        class_mode = None

    # Image prediction model.
    if enet_state is not None:
        model = MultiViewDLElasticNet(*enet_state)
    elif dl_state is not None:
        model = load_model(dl_state)
    else:
        raise Exception('At least one image model state must be specified.')

    # XGB model.
    xgb_clf = pickle.load(open(xgb_state))

    # Print header.
    fout = open(out_pred, 'w')
    if validation_mode:
        fout.write(dminfer.INFER_HEADER_VAL)
    else:
        fout.write(dminfer.INFER_HEADER)

    # Loop through all last 2 exam pairs.
    for subj_id, curr_idx, curr_dat, prior_idx, prior_dat in last2_exgen:
        # Get meta info for both breasts.
        left_record, right_record = meta_man.get_info_exam_pair(
            curr_dat, prior_dat)
        nb_days = left_record['daysSincePreviousExam']

        # Get image data and make predictions.
        exam_list = []
        exam_list.append( (subj_id, curr_idx, 
                           meta_man.get_info_per_exam(curr_dat)) )
        if prior_idx is not None:
            exam_list.append( (subj_id, prior_idx, 
                               meta_man.get_info_per_exam(prior_dat)) )
        datgen_exam = img_gen.flow_from_exam_list(
            exam_list, target_size=(img_size[0], img_size[1]), 
            class_mode=class_mode, prediction_mode=True, 
            batch_size=len(exam_list), verbose=False)
        ebat = next(datgen_exam)
        if class_mode is not None:
            bat_x = ebat[0]
            bat_y = ebat[1]
        else:
            bat_x = ebat
        cc_batch = bat_x[2]
        mlo_batch = bat_x[3]
        curr_left_score = dminfer.pred_2view_img_list(
            cc_batch[0], mlo_batch[0], model, use_mean)
        curr_right_score = dminfer.pred_2view_img_list(
            cc_batch[1], mlo_batch[1], model, use_mean)
        if prior_idx is not None:
            prior_left_score = dminfer.pred_2view_img_list(
                cc_batch[2], mlo_batch[2], model, use_mean)
            prior_right_score = dminfer.pred_2view_img_list(
                cc_batch[3], mlo_batch[3], model, use_mean)
            diff_left_score = (curr_left_score - prior_left_score)/nb_days*365
            diff_right_score = (curr_right_score - prior_right_score)/nb_days*365
        else:
            prior_left_score = np.nan
            prior_right_score = np.nan
            diff_left_score = np.nan
            diff_right_score = np.nan

        # Merge image scores into meta info.
        left_record = left_record\
                .assign(curr_score=curr_left_score)\
                .assign(prior_score=prior_left_score)\
                .assign(diff_score=diff_left_score)
        right_record = right_record\
                .assign(curr_score=curr_right_score)\
                .assign(prior_score=prior_right_score)\
                .assign(diff_score=diff_right_score)
        dsubj = xgb.DMatrix(pd.concat([left_record, right_record], 
                                      ignore_index=True))

        # Predict using XGB.
        pred = xgb_clf.predict(dsubj, ntree_limit=xgb_clf.best_ntree_limit)

        # Output.
        if validation_mode:
            fout.write("%s\t%s\tL\t%f\t%f\n" % \
                       (str(subj_id), str(curr_idx), pred[0], bat_y[0]))
            fout.write("%s\t%s\tR\t%f\t%f\n" % \
                       (str(subj_id), str(curr_idx), pred[1], bat_y[1]))
        else:
            fout.write("%s\tL\t%f\n" % (str(subj_id), pred[0]))
            fout.write("%s\tR\t%f\n" % (str(subj_id), pred[1]))


    fout.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="DM SC2 inference")
    parser.add_argument("img_folder", type=str)
    parser.add_argument("--img-size", "-is", dest="img_size", nargs=2, type=int, 
                        default=[288, 224])
    parser.add_argument("--featurewise-norm", dest="do_featurewise_norm", action="store_true")
    parser.add_argument("--no-featurewise-norm", dest="do_featurewise_norm", action="store_false")
    parser.set_defaults(do_featurewise_norm=True)
    parser.add_argument("--featurewise-mean", "-feam", dest="featurewise_mean", 
                        type=float, default=485.9)
    parser.add_argument("--featurewise-std", "-feas", dest="featurewise_std", 
                        type=float, default=765.2)
    parser.add_argument("--img-tsv", "-it", dest="img_tsv", type=str, 
                        default="./metadata/images_crosswalk.tsv")
    parser.add_argument("--exam-tsv", "-et", dest="exam_tsv", type=str)
    parser.add_argument("--no-exam-tsv", dest="exam_tsv", action="store_const", const=None)
    parser.set_defaults(exam_tsv=None)
    parser.add_argument("--dl-state", "-ds", dest="dl_state", type=str)
    parser.add_argument("--enet-state", "-es", dest="enet_state", nargs=2, type=str)
    parser.add_argument("--xgb-state", "-xs", dest="xgb_state", type=str)
    parser.add_argument("--validation-mode", dest="validation_mode", action="store_true")
    parser.add_argument("--no-validation-mode", dest="validation_mode", action="store_false")
    parser.set_defaults(validation_mode=False)
    parser.add_argument("--use-mean", dest="use_mean", action="store_true")
    parser.add_argument("--no-use-mean", dest="use_mean", action="store_false")
    parser.set_defaults(use_mean=False)
    parser.add_argument("--out-pred", "-o", dest="out_pred", type=str, 
                        default="./output/predictions.tsv")

    args = parser.parse_args()
    run_opts = dict(
        img_size=args.img_size, 
        do_featurewise_norm=args.do_featurewise_norm,
        featurewise_mean=args.featurewise_mean,
        featurewise_std=args.featurewise_std,
        img_tsv=args.img_tsv,
        exam_tsv=args.exam_tsv,
        dl_state=args.dl_state,
        enet_state=args.enet_state,
        xgb_state=args.xgb_state,
        validation_mode=args.validation_mode,
        use_mean=args.use_mean,
        out_pred=args.out_pred
    )
    print "\n>>> Inference options: <<<\n", run_opts, "\n"
    run(args.img_folder, **run_opts)

