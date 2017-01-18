import argparse
import numpy as np
from keras.models import load_model
from meta import DMMetaManager
from dm_image import DMImageDataGenerator

import warnings
import exceptions
warnings.filterwarnings('ignore', category=exceptions.UserWarning)


def run(img_folder, img_size=[288, 224], do_featurewise_norm=True, 
        featurewise_mean=485.9, featurewise_std=765.2, batch_size=16, 
        img_tsv='./metadata/images_crosswalk_prediction.tsv',
        saved_state='./modelState/dm_resnet_best_model.h5',
        validation_mode=False, use_mean=False,
        out_pred='./output/predictions.tsv'):
    '''Run SC1 inference
    Args:
        featurewise_mean, featurewise_std ([float]): they are estimated from 
                1152 x 896 images. Using different sized images give very close
                results. For png, mean=7772, std=12187.
    '''

    # Setup data generator for inference.
    meta_man = DMMetaManager(
        img_tsv=img_tsv, exam_tsv="", img_folder=img_folder, 
        img_extension='dcm')
    last_exam_list = meta_man.get_last_exam_list()
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
    datgen_exam = img_gen.flow_from_exam_list(
        last_exam_list, target_size=(img_size[0], img_size[1]), 
        class_mode=class_mode, prediction_mode=True, batch_size=batch_size)


    def pred_2view_img_list(cc_img_list, mlo_img_list, model):
        '''Make predictions for all pairwise combinations of the 2 views
        '''
        pass
        pred_cc_list = []
        pred_mlo_list = []
        for cc in cc_img_list:
            for mlo in mlo_img_list:
                pred_cc_list.append(cc)
                pred_mlo_list.append(mlo)
        pred_cc = np.stack(pred_cc_list)
        pred_mlo = np.stack(pred_mlo_list)
        preds = model.predict_on_batch([pred_cc, pred_mlo])
        return preds

    model = load_model(saved_state)
    exams_seen = 0
    fout = open('./output/predictions.tsv', 'w')

    # Print header.
    if class_mode is not None:
        fout.write("subjectId\tlaterality\tconfidence\ttarget\n")
    else:
        fout.write("subjectId\tlaterality\tconfidence\n")

    while exams_seen < len(last_exam_list):
        ebat = next(datgen_exam)
        if class_mode is not None:
            bat_x = ebat[0]
            bat_y = ebat[1]
        else:
            bat_x = ebat
        subj_list = bat_x[0]
        cc_list = bat_x[1]
        mlo_list = bat_x[2]
        for i, subj in enumerate(subj_list):
            li = i*2        # left breast index.
            ri = i*2 + 1    # right breast index.
            left_preds = pred_2view_img_list(cc_list[li], mlo_list[li], model)
            right_preds = pred_2view_img_list(cc_list[ri], mlo_list[ri], model)
            if not use_mean:
                left_pred = np.array(left_preds).max()
                right_pred = np.array(right_preds).max()
            else:
                left_pred = np.array(left_preds).mean()
                right_pred = np.array(right_preds).mean()
            if class_mode is not None:
                fout.write("%s\tL\t%f\t%f\n" % (str(subj), left_pred, bat_y[li]))
                fout.write("%s\tR\t%f\t%f\n" % (str(subj), right_pred, bat_y[ri]))
            else:
                fout.write("%s\tL\t%f\n" % (str(subj), left_pred))
                fout.write("%s\tR\t%f\n" % (str(subj), right_pred))

        exams_seen += len(subj_list)

    fout.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="DM SC1 inference")
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
    parser.add_argument("--batch-size", "-bs", dest="batch_size", type=int, default=16)
    parser.add_argument("--img-tsv", "-it", dest="img_tsv", type=str, 
                        default="./metadata/images_crosswalk.tsv")
    parser.add_argument("--saved-state", "-ss", dest="saved_state", type=str, 
                        default="./modelState/dm_resnet_best_model.h5")
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
        batch_size=args.batch_size, 
        img_tsv=args.img_tsv,
        saved_state=args.saved_state,
        validation_mode=args.validation_mode,
        use_mean=args.use_mean,
        out_pred=args.out_pred
    )
    run(args.img_folder, **run_opts)

