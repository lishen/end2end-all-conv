import argparse, os
import numpy as np
from keras.models import load_model
from meta import DMMetaManager
from dm_image import DMImageDataGenerator
import dm_inference as dminfer
from dm_keras_ext import DMMetrics
from dm_multi_gpu import make_parallel

import warnings
import exceptions
warnings.filterwarnings('ignore', category=exceptions.UserWarning)


def run(img_folder, img_height=1024, img_scale=4095, 
        roi_per_img=32, roi_size=(256, 256), 
        low_int_threshold=.05, blob_min_area=3, 
        blob_min_int=.5, blob_max_int=.85, blob_th_step=10,
        roi_state=None, roi_bs=32,
        do_featurewise_norm=True, featurewise_mean=884.7, featurewise_std=745.3, 
        img_tsv='./metadata/images_crosswalk_prediction.tsv', exam_tsv=None,
        dl_state=None, dl_bs=32, nb_top_avg=1, validation_mode=False,
        img_voting=False,
        out_pred='./output/predictions.tsv'):
    '''Run SC1 inference using the candidate ROI approach
    Notes: 
        "mean=884.7, std=745.3" are estimated from 20 subjects on the 
        training data.
    '''

    # Read some env variables.
    random_seed = int(os.getenv('RANDOM_SEED', 12345))
    # nb_worker = int(os.getenv('NUM_CPU_CORES', 4))
    gpu_count = int(os.getenv('NUM_GPU_DEVICES', 1))

    # Setup data generator for inference.
    meta_man = DMMetaManager(
        img_tsv=img_tsv, exam_tsv=exam_tsv, img_folder=img_folder, 
        img_extension='dcm')
    if validation_mode:
        exam_list = meta_man.get_flatten_exam_list(flatten_img_list=True)
    else:
        exam_list = meta_man.get_last_exam_list(flatten_img_list=True)
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

    # Load ROI classifier.
    if roi_state is not None:
        roi_clf = load_model(
            roi_state, 
            custom_objects={
                'sensitivity': DMMetrics.sensitivity, 
                'specificity': DMMetrics.specificity
            }
        )
        if gpu_count > 1:
            roi_clf = make_parallel(roi_clf, gpu_count)
    else:
        roi_clf = None

    # Load model.
    if dl_state is not None:
        model = load_model(
            dl_state, 
            custom_objects={
                'sensitivity': DMMetrics.sensitivity, 
                'specificity': DMMetrics.specificity
            }
        )
    else:
        raise Exception('At least one model state must be specified.')
    if gpu_count > 1:
        model = make_parallel(model, gpu_count)
    
    # A function to make predictions on image patches from an image list.
    def pred_img_list(img_list):
        roi_generator = img_gen.flow_from_candid_roi(
            img_list, target_height=img_height, target_scale=img_scale,
            validation_mode=True, 
            img_per_batch=len(img_list), roi_per_img=roi_per_img, 
            roi_size=roi_size,
            low_int_threshold=low_int_threshold, blob_min_area=blob_min_area,
            blob_min_int=blob_min_int, blob_max_int=blob_max_int, 
            blob_th_step=blob_th_step, 
            roi_clf=roi_clf, clf_bs=roi_bs, return_sample_weight=True,
            seed=random_seed)
        roi_dat, roi_w = roi_generator.next()
        # import pdb; pdb.set_trace()
        pred = model.predict(roi_dat, batch_size=dl_bs).ravel()
        if roi_clf is not None:
            # return np.average(pred, weights=roi_w)
            # import pdb; pdb.set_trace()
            return pred[np.argmax(roi_w)]
        elif img_voting:
            pred = pred.reshape((-1, roi_per_img))
            img_preds = [ np.sort(row)[-nb_top_avg:].mean() for row in pred ]
            return np.mean(img_preds)
        else:
            return np.sort(pred)[-nb_top_avg:].mean()


    # Print header.
    fout = open(out_pred, 'w')
    if validation_mode:
        fout.write(dminfer.INFER_HEADER_VAL)
    else:
        fout.write(dminfer.INFER_HEADER)

    for subj, exidx, exam in exam_list:
        try:
            predL = pred_img_list(exam['L']['img'])
        except KeyError:
            predL = .0
        try:
            predR = pred_img_list(exam['R']['img'])
        except KeyError:
            predR = .0

        try:
            cancerL = int(exam['L']['cancer'])
        except ValueError:
            cancerL = 0
        try:
            cancerR = int(exam['R']['cancer'])
        except ValueError:
            cancerR = 0

        if validation_mode:
            fout.write("%s\t%s\tL\t%f\t%d\n" % \
                       (str(subj), str(exidx), predL, cancerL))
            fout.write("%s\t%s\tR\t%f\t%d\n" % \
                       (str(subj), str(exidx), predR, cancerR))
        else:
            fout.write("%s\tL\t%f\n" % (str(subj), predL))
            fout.write("%s\tR\t%f\n" % (str(subj), predR))

    fout.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="DM SC1 inference")
    parser.add_argument("img_folder", type=str)
    parser.add_argument("--img-height", "-ih", dest="img_height", type=int, default=1024)
    parser.add_argument("--img-scale", "-is", dest="img_scale", type=int, default=4095)
    parser.add_argument("--roi-per-img", "-rpi", dest="roi_per_img", type=int, default=32)
    parser.add_argument("--roi-size", "-rs", dest="roi_size", nargs=2, type=int, default=[256, 256])
    parser.add_argument("--low-int-threshold", dest="low_int_threshold", type=float, default=.05)
    parser.add_argument("--blob-min-area", dest="blob_min_area", type=int, default=3)
    parser.add_argument("--blob-min-int", dest="blob_min_int", type=float, default=.5)
    parser.add_argument("--blob-max-int", dest="blob_max_int", type=float, default=.85)
    parser.add_argument("--blob-th-step", dest="blob_th_step", type=int, default=10)
    parser.add_argument("--roi-state", dest="roi_state", type=str, default=None)
    parser.add_argument("--no-roi-state", dest="roi_state", action="store_const", const=None)
    parser.add_argument("--roi-bs", dest="roi_bs", type=int, default=32)
    parser.add_argument("--featurewise-norm", dest="do_featurewise_norm", action="store_true")
    parser.add_argument("--no-featurewise-norm", dest="do_featurewise_norm", action="store_false")
    parser.set_defaults(do_featurewise_norm=True)
    parser.add_argument("--featurewise-mean", "-feam", dest="featurewise_mean", type=float, default=884.7)
    parser.add_argument("--featurewise-std", "-feas", dest="featurewise_std", type=float, default=745.3)
    parser.add_argument("--img-tsv", "-it", dest="img_tsv", type=str, 
                        default="./metadata/images_crosswalk.tsv")
    parser.add_argument("--exam-tsv", "-et", dest="exam_tsv", type=str)
    parser.add_argument("--no-exam-tsv", dest="exam_tsv", action="store_const", const=None)
    parser.set_defaults(exam_tsv=None)
    parser.add_argument("--dl-state", "-ds", dest="dl_state", type=str)
    parser.add_argument("--dl-bs", "-bs", dest="dl_bs", type=int, default=32)
    parser.add_argument("--nb-top-avg", dest="nb_top_avg", type=int, default=1)
    parser.add_argument("--validation-mode", dest="validation_mode", action="store_true")
    parser.add_argument("--no-validation-mode", dest="validation_mode", action="store_false")
    parser.set_defaults(validation_mode=False)
    parser.add_argument("--img-voting", dest="img_voting", action="store_true")
    parser.add_argument("--no-img-voting", dest="img_voting", action="store_false")
    parser.set_defaults(img_voting=False)
    parser.add_argument("--out-pred", "-o", dest="out_pred", type=str, 
                        default="./output/predictions.tsv")

    args = parser.parse_args()
    run_opts = dict(
        img_height=args.img_height, 
        img_scale=args.img_scale,
        roi_per_img=args.roi_per_img,
        roi_size=tuple(args.roi_size),
        low_int_threshold=args.low_int_threshold,
        blob_min_area=args.blob_min_area,
        blob_min_int=args.blob_min_int,
        blob_max_int=args.blob_max_int,
        blob_th_step=args.blob_th_step,
        roi_state=args.roi_state,
        roi_bs=args.roi_bs,
        do_featurewise_norm=args.do_featurewise_norm,
        featurewise_mean=args.featurewise_mean,
        featurewise_std=args.featurewise_std,
        img_tsv=args.img_tsv,
        exam_tsv=args.exam_tsv,
        dl_state=args.dl_state,
        dl_bs=args.dl_bs,
        nb_top_avg=args.nb_top_avg,
        validation_mode=args.validation_mode,
        img_voting=args.img_voting,
        out_pred=args.out_pred
    )
    print "\n>>> Inference options: <<<\n", run_opts, "\n"
    run(args.img_folder, **run_opts)

