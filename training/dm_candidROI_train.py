import os, argparse, sys
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import (
    ReduceLROnPlateau, 
    EarlyStopping, 
)
from keras.optimizers import SGD
from keras.models import load_model
import tensorflow as tf
from meta import DMMetaManager
from dm_image import DMImageDataGenerator, to_sparse
from dm_resnet import ResNetBuilder
from dm_multi_gpu import make_parallel
from dm_keras_ext import DMMetrics, DMAucModelCheckpoint

import warnings
import exceptions
warnings.filterwarnings('ignore', category=exceptions.UserWarning)


def subset_img_labs(img_list, lab_list, neg_vs_pos_ratio, seed=12345):
    rng = np.random.RandomState(seed)
    img_list = np.array(img_list)
    lab_list = np.array(lab_list)
    pos_idx = np.where(lab_list==1)[0]
    neg_idx = np.where(lab_list==0)[0]
    nb_neg_desired = int(len(pos_idx)*neg_vs_pos_ratio)
    if nb_neg_desired < len(neg_idx):
        sampled_neg_idx = rng.choice(neg_idx, nb_neg_desired, replace=False)
        all_idx = np.concatenate([pos_idx, sampled_neg_idx])
        img_list = img_list[all_idx].tolist()
        lab_list = lab_list[all_idx].tolist()
        return img_list, lab_list
    else:
        return img_list.tolist(), lab_list.tolist()


def load_dat_ram(generator, nb_samples):
    samples_seen = 0
    X_list = []
    y_list = []
    w_list = []
    while samples_seen < nb_samples:
        blob_ = generator.next()
        try:
            X,y,w = blob_
            w_list.append(w)
        except ValueError:
            X,y = blob_
        X_list.append(X)
        y_list.append(y)
        samples_seen += len(y)
    try:
        data_set = (np.concatenate(X_list), 
                    np.concatenate(y_list),
                    np.concatenate(w_list))
    except ValueError:
        data_set = (np.concatenate(X_list), 
                    np.concatenate(y_list))

    if len(data_set[0]) != nb_samples:
        raise Exception('Load data into RAM error')

    return data_set


def run(img_folder, img_extension='dcm', 
        img_height=1024, img_scale=4095, 
        do_featurewise_norm=True, norm_fit_size=10,
        img_per_batch=2, roi_per_img=32, roi_size=(256, 256), 
        one_patch_mode=False,
        low_int_threshold=.05, blob_min_area=3, 
        blob_min_int=.5, blob_max_int=.85, blob_th_step=10,
        more_augmentation=False, roi_state=None, clf_bs=32, cutpoint=.5,
        amp_factor=1., return_sample_weight=True, auto_batch_balance=True,
        patches_per_epoch=12800, nb_epoch=20, 
        train_neg_vs_pos_ratio=None, all_neg_skip=0., 
        nb_init_filter=32, init_filter_size=5, init_conv_stride=2, 
        pool_size=2, pool_stride=2, 
        weight_decay=.0001, alpha=.0001, l1_ratio=.0, 
        inp_dropout=.0, hidden_dropout=.0, init_lr=.01,
        val_size=.2, val_neg_vs_pos_ratio=None, lr_patience=3, es_patience=10, 
        resume_from=None, net='resnet50', load_val_ram=False, 
        load_train_ram=False, no_pos_skip=0.,
        exam_tsv='./metadata/exams_metadata.tsv',
        img_tsv='./metadata/images_crosswalk.tsv',
        best_model='./modelState/dm_candidROI_best_model.h5',
        final_model="NOSAVE"):
    '''Run ResNet training on candidate ROIs from mammograms
    Args:
        norm_fit_size ([int]): the number of patients used to calculate 
                feature-wise mean and std.
    '''

    # Read some env variables.
    random_seed = int(os.getenv('RANDOM_SEED', 12345))
    # Use of multiple CPU cores is not working!
    # When nb_worker>1 and pickle_safe=True, this error is encountered:
    # "failed to enqueue async memcpy from host to device: CUDA_ERROR_NOT_INITIALIZED"
    # To avoid the error, only this combination worked: 
    # nb_worker=1 and pickle_safe=False.
    nb_worker = int(os.getenv('NUM_CPU_CORES', 4))
    gpu_count = int(os.getenv('NUM_GPU_DEVICES', 1))
    
    # Setup training and validation data.
    # Load image or exam lists and split them into train and val sets.
    meta_man = DMMetaManager(exam_tsv=exam_tsv, 
                             img_tsv=img_tsv, 
                             img_folder=img_folder, 
                             img_extension=img_extension)
    # Split data based on subjects.
    subj_list, subj_labs = meta_man.get_subj_labs()
    if val_size < 0:  # do not split data.
        subj_train = subj_test = subj_list
    else:
        subj_train, subj_test = train_test_split(
            subj_list, test_size=val_size, random_state=random_seed, 
            stratify=subj_labs)
    img_train, lab_train = meta_man.get_flatten_img_list(subj_train)
    img_test, lab_test = meta_man.get_flatten_img_list(subj_test)

    # Sample data sets to desired ratio.
    if train_neg_vs_pos_ratio is not None:
        all_neg_skip = .0  # turn off batch-level sampling.
        img_train, lab_train = subset_img_labs(
            img_train, lab_train, train_neg_vs_pos_ratio, random_seed)
    if val_neg_vs_pos_ratio is not None:
        img_test, lab_test = subset_img_labs(
            img_test, lab_test, val_neg_vs_pos_ratio, random_seed)

    # Create image generators for train, fit and val.
    imgen_trainval = DMImageDataGenerator(
        horizontal_flip=True, 
        vertical_flip=True)
    if more_augmentation:
        imgen_trainval.rotation_range = 90
        imgen_trainval.width_shift_range = .25
        imgen_trainval.height_shift_range = .25
        imgen_trainval.zoom_range = [.8, 1.2]

    if do_featurewise_norm:
        imgen_trainval.featurewise_center = True
        imgen_trainval.featurewise_std_normalization = True
        # Fit feature-wise mean and std.
        img_fit,_ = meta_man.get_flatten_img_list(
            subj_train[:norm_fit_size])  # fit on a subset.
        print ">>> Fit image generator <<<"; sys.stdout.flush()
        fit_generator = imgen_trainval.flow_from_candid_roi(
            img_fit,
            target_height=img_height, target_scale=img_scale,
            class_mode=None, validation_mode=True, 
            img_per_batch=len(img_fit), roi_per_img=roi_per_img, 
            roi_size=roi_size,
            low_int_threshold=low_int_threshold, blob_min_area=blob_min_area, 
            blob_min_int=blob_min_int, blob_max_int=blob_max_int, 
            blob_th_step=blob_th_step,
            roi_clf=None, return_sample_weight=False, seed=random_seed)
        imgen_trainval.fit(fit_generator.next())
        print "Estimates from %d images: mean=%.1f, std=%.1f." % \
            (len(img_fit), imgen_trainval.mean, imgen_trainval.std)
        sys.stdout.flush()
    else:
        imgen_trainval.samplewise_center = True
        imgen_trainval.samplewise_std_normalization = True

    # Load ROI classifier.
    if roi_state is not None:
        roi_clf = load_model(
            roi_state, 
            custom_objects={
                'sensitivity': DMMetrics.sensitivity, 
                'specificity': DMMetrics.specificity
            }
        )
        graph = tf.get_default_graph()
    else:
        roi_clf = None
        graph = None

    # Set some DL training related parameters.
    if one_patch_mode:
        class_mode = 'binary'
        loss = 'binary_crossentropy'
        metrics = [DMMetrics.sensitivity, DMMetrics.specificity]
    else:
        class_mode = 'categorical'
        loss = 'categorical_crossentropy'
        metrics = ['accuracy', 'precision', 'recall']
    if load_train_ram:
        validation_mode = True
        return_raw_img = True
    else:
        validation_mode = False
        return_raw_img = False

    # Create train and val generators.
    print ">>> Train image generator <<<"; sys.stdout.flush()
    train_generator = imgen_trainval.flow_from_candid_roi(
        img_train, lab_train, 
        target_height=img_height, target_scale=img_scale,
        class_mode=class_mode, validation_mode=validation_mode, 
        img_per_batch=img_per_batch, roi_per_img=roi_per_img, 
        roi_size=roi_size, one_patch_mode=one_patch_mode,
        low_int_threshold=low_int_threshold, blob_min_area=blob_min_area, 
        blob_min_int=blob_min_int, blob_max_int=blob_max_int, 
        blob_th_step=blob_th_step,
        tf_graph=graph, roi_clf=roi_clf, clf_bs=clf_bs, cutpoint=cutpoint,
        amp_factor=amp_factor, return_sample_weight=return_sample_weight,
        auto_batch_balance=auto_batch_balance,
        all_neg_skip=all_neg_skip, shuffle=True, seed=random_seed,
        return_raw_img=return_raw_img)

    print ">>> Validation image generator <<<"; sys.stdout.flush()
    val_generator = imgen_trainval.flow_from_candid_roi(
        img_test, lab_test, 
        target_height=img_height, target_scale=img_scale,
        class_mode=class_mode, validation_mode=True, 
        img_per_batch=img_per_batch, roi_per_img=roi_per_img, 
        roi_size=roi_size, one_patch_mode=one_patch_mode,
        low_int_threshold=low_int_threshold, blob_min_area=blob_min_area, 
        blob_min_int=blob_min_int, blob_max_int=blob_max_int, 
        blob_th_step=blob_th_step,
        tf_graph=graph, roi_clf=roi_clf, clf_bs=clf_bs, cutpoint=cutpoint,
        amp_factor=amp_factor, return_sample_weight=False, 
        auto_batch_balance=False,
        seed=random_seed)

    # Load train and validation set into RAM.
    if one_patch_mode:
        nb_train_samples = len(img_train)
        nb_val_samples = len(img_test)
    else:
        nb_train_samples = len(img_train)*roi_per_img
        nb_val_samples = len(img_test)*roi_per_img
    if load_val_ram:
        print "Loading validation data into RAM.",
        sys.stdout.flush()
        validation_set = load_dat_ram(val_generator, nb_val_samples)
        print "Done."; sys.stdout.flush()
        sparse_y = to_sparse(validation_set[1])
        for uy in np.unique(sparse_y):
            print "Nb of samples for class:%d = %d" % \
                    (uy, (sparse_y==uy).sum())
        sys.stdout.flush()
    if load_train_ram:
        print "Loading train data into RAM.",
        sys.stdout.flush()
        train_set = load_dat_ram(train_generator, nb_train_samples)
        print "Done."; sys.stdout.flush()
        sparse_y = to_sparse(train_set[1])
        for uy in np.unique(sparse_y):
            print "Nb of samples for class:%d = %d" % \
                    (uy, (sparse_y==uy).sum())
        sys.stdout.flush()
        train_generator = imgen_trainval.flow(
            train_set[0], train_set[1], batch_size=clf_bs, 
            auto_batch_balance=auto_batch_balance, no_pos_skip=no_pos_skip,
            shuffle=True, seed=random_seed)

    # Load or create model.
    if resume_from is not None:
        model = load_model(
            resume_from,
            custom_objects={
                'sensitivity': DMMetrics.sensitivity, 
                'specificity': DMMetrics.specificity
            }
        )
    else:
        builder = ResNetBuilder
        if net == 'resnet18':
            model = builder.build_resnet_18(
                (1, roi_size[0], roi_size[1]), 3, nb_init_filter, init_filter_size, 
                init_conv_stride, pool_size, pool_stride, weight_decay, alpha, l1_ratio, 
                inp_dropout, hidden_dropout)
        elif net == 'resnet34':
            model = builder.build_resnet_34(
                (1, roi_size[0], roi_size[1]), 3, nb_init_filter, init_filter_size, 
                init_conv_stride, pool_size, pool_stride, weight_decay, alpha, l1_ratio, 
                inp_dropout, hidden_dropout)
        elif net == 'resnet50':
            model = builder.build_resnet_50(
                (1, roi_size[0], roi_size[1]), 3, nb_init_filter, init_filter_size, 
                init_conv_stride, pool_size, pool_stride, weight_decay, alpha, l1_ratio, 
                inp_dropout, hidden_dropout)
        elif net == 'resnet101':
            model = builder.build_resnet_101(
                (1, roi_size[0], roi_size[1]), 3, nb_init_filter, init_filter_size, 
                init_conv_stride, pool_size, pool_stride, weight_decay, alpha, l1_ratio, 
                inp_dropout, hidden_dropout)
        elif net == 'resnet152':
            model = builder.build_resnet_152(
                (1, roi_size[0], roi_size[1]), 3, nb_init_filter, init_filter_size, 
                init_conv_stride, pool_size, pool_stride, weight_decay, alpha, l1_ratio, 
                inp_dropout, hidden_dropout)
    
    if gpu_count > 1:
        model = make_parallel(model, gpu_count)

    # Model training.
    sgd = SGD(lr=init_lr, momentum=0.9, decay=0.0, nesterov=True)
    model.compile(optimizer=sgd, loss=loss, metrics=metrics)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, 
                                  patience=lr_patience, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=es_patience, 
                                   verbose=1)
    if load_val_ram:
        auc_checkpointer = DMAucModelCheckpoint(
            best_model, validation_set, batch_size=clf_bs)
    else:
        auc_checkpointer = DMAucModelCheckpoint(
            best_model, val_generator, nb_test_samples=nb_val_samples)
    hist = model.fit_generator(
        train_generator, 
        samples_per_epoch=patches_per_epoch, 
        nb_epoch=nb_epoch,
        validation_data=validation_set if load_val_ram else val_generator, 
        nb_val_samples=nb_val_samples, 
        callbacks=[reduce_lr, early_stopping, auc_checkpointer],
        nb_worker=1, pickle_safe=False,
        # nb_worker=nb_worker if load_train_ram else 1,
        # pickle_safe=True if load_train_ram else False,
        verbose=2)

    if final_model != "NOSAVE":
        print "Saving final model to:", final_model; sys.stdout.flush()
        model.save(final_model)
    
    # Training report.
    min_loss_locs, = np.where(hist.history['val_loss'] == min(hist.history['val_loss']))
    best_val_loss = hist.history['val_loss'][min_loss_locs[0]]
    if one_patch_mode:
        best_val_sensitivity = hist.history['val_sensitivity'][min_loss_locs[0]]
        best_val_specificity = hist.history['val_specificity'][min_loss_locs[0]]
    else:
        best_val_precision = hist.history['val_precision'][min_loss_locs[0]]
        best_val_recall = hist.history['val_recall'][min_loss_locs[0]]
        best_val_accuracy = hist.history['val_acc'][min_loss_locs[0]]
    print "\n==== Training summary ===="
    print "Minimum val loss achieved at epoch:", min_loss_locs[0] + 1
    print "Best val loss:", best_val_loss
    if one_patch_mode:
        print "Best val sensitivity:", best_val_sensitivity
        print "Best val specificity:", best_val_specificity
    else:
        print "Best val precision:", best_val_precision
        print "Best val recall:", best_val_recall
        print "Best val accuracy:", best_val_accuracy

    return hist


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="DM Candid ROI training")
    parser.add_argument("img_folder", type=str)
    parser.add_argument("--img-extension", "-ext", dest="img_extension", type=str, default="dcm")
    parser.add_argument("--img-height", "-ih", dest="img_height", type=int, default=1024)
    parser.add_argument("--img-scale", "-ic", dest="img_scale", type=int, default=4095)
    parser.add_argument("--featurewise-norm", dest="do_featurewise_norm", action="store_true")
    parser.add_argument("--no-featurewise-norm", dest="do_featurewise_norm", action="store_false")
    parser.set_defaults(do_featurewise_norm=True)
    parser.add_argument("--norm-fit-size", "-nfs", dest="norm_fit_size", type=int, default=10)
    parser.add_argument("--img-per-batch", "-ipb", dest="img_per_batch", type=int, default=2)
    parser.add_argument("--roi-per-img", "-rpi", dest="roi_per_img", type=int, default=32)
    parser.add_argument("--roi-size", dest="roi_size", nargs=2, type=int, default=[256, 256])
    parser.add_argument("--one-patch-mode", dest="one_patch_mode", action="store_true")
    parser.add_argument("--no-one-patch-mode", dest="one_patch_mode", action="store_false")
    parser.set_defaults(one_patch_mode=False)
    parser.add_argument("--low-int-threshold", dest="low_int_threshold", type=float, default=.05)
    parser.add_argument("--blob-min-area", dest="blob_min_area", type=int, default=3)
    parser.add_argument("--blob-min-int", dest="blob_min_int", type=float, default=.5)
    parser.add_argument("--blob-max-int", dest="blob_max_int", type=float, default=.85)
    parser.add_argument("--blob-th-step", dest="blob_th_step", type=int, default=10)
    parser.add_argument("--more-augmentation", dest="more_augmentation", action="store_true")
    parser.add_argument("--no-more-augmentation", dest="more_augmentation", action="store_false")
    parser.set_defaults(more_augmentation=False)
    parser.add_argument("--roi-state", dest="roi_state", type=str, default=None)
    parser.add_argument("--no-roi-state", dest="roi_state", action="store_const", const=None)
    parser.add_argument("--clf-bs", dest="clf_bs", type=int, default=32)
    parser.add_argument("--cutpoint", dest="cutpoint", type=float, default=.5)
    parser.add_argument("--amp-factor", dest="amp_factor", type=float, default=1.)
    parser.add_argument("--return-sample-weight", dest="return_sample_weight", action="store_true")
    parser.add_argument("--no-return-sample-weight", dest="return_sample_weight", action="store_false")
    parser.set_defaults(return_sample_weight=True)
    parser.add_argument("--auto-batch-balance", dest="auto_batch_balance", action="store_true")
    parser.add_argument("--no-auto-batch-balance", dest="auto_batch_balance", action="store_false")
    parser.set_defaults(auto_batch_balance=True)
    parser.add_argument("--patches-per-epoch", "-ppe", dest="patches_per_epoch", type=int, default=12800)
    parser.add_argument("--nb-epoch", "-ne", dest="nb_epoch", type=int, default=20)
    parser.add_argument("--train-nvp-ratio", dest="train_neg_vs_pos_ratio", type=float, default=None)
    parser.add_argument("--no-train-nvp-ratio", dest="train_neg_vs_pos_ratio", action="store_const", const=None)
    parser.add_argument("--allneg-skip", dest="all_neg_skip", type=float, default=0.)
    parser.add_argument("--nopos-skip", dest="no_pos_skip", type=float, default=0.)
    parser.add_argument("--nb-init-filter", "-nif", dest="nb_init_filter", type=int, default=32)
    parser.add_argument("--init-filter-size", "-ifs", dest="init_filter_size", type=int, default=5)
    parser.add_argument("--init-conv-stride", "-ics", dest="init_conv_stride", type=int, default=2)
    parser.add_argument("--max-pooling-size", "-mps", dest="pool_size", type=int, default=2)
    parser.add_argument("--max-pooling-stride", "-mpr", dest="pool_stride", type=int, default=2)
    parser.add_argument("--weight-decay", "-wd", dest="weight_decay", type=float, default=.0001)
    parser.add_argument("--alpha", dest="alpha", type=float, default=.0001)
    parser.add_argument("--l1-ratio", dest="l1_ratio", type=float, default=.0)
    parser.add_argument("--inp-dropout", "-id", dest="inp_dropout", type=float, default=.0)
    parser.add_argument("--hidden-dropout", "-hd", dest="hidden_dropout", type=float, default=.0)
    parser.add_argument("--init-learningrate", "-ilr", dest="init_lr", type=float, default=.01)
    parser.add_argument("--val-size", "-vs", dest="val_size", type=float, default=.2)
    parser.add_argument("--val-nvp-ratio", dest="val_neg_vs_pos_ratio", type=float, default=None)
    parser.add_argument("--no-val-nvp-ratio", dest="val_neg_vs_pos_ratio", action="store_const", const=None)
    parser.add_argument("--lr-patience", "-lrp", dest="lr_patience", type=int, default=3)
    parser.add_argument("--es-patience", "-esp", dest="es_patience", type=int, default=10)
    parser.add_argument("--resume-from", "-rf", dest="resume_from", type=str, default=None)
    parser.add_argument("--no-resume-from", "-nrf", dest="resume_from", action="store_const", const=None)
    parser.add_argument("--net", dest="net", type=str, default="resnet50")
    parser.add_argument("--loadval-ram", dest="load_val_ram", action="store_true")
    parser.add_argument("--no-loadval-ram", dest="load_val_ram", action="store_false")
    parser.set_defaults(load_val_ram=False)
    parser.add_argument("--loadtrain-ram", dest="load_train_ram", action="store_true")
    parser.add_argument("--no-loadtrain-ram", dest="load_train_ram", action="store_false")
    parser.set_defaults(load_train_ram=False)
    parser.add_argument("--exam-tsv", "-et", dest="exam_tsv", type=str, 
                        default="./metadata/exams_metadata.tsv")
    parser.add_argument("--no-exam-tsv", dest="exam_tsv", action="store_const", const=None)
    parser.add_argument("--img-tsv", "-it", dest="img_tsv", type=str, 
                        default="./metadata/images_crosswalk.tsv")
    parser.add_argument("--best-model", "-bm", dest="best_model", type=str, 
                        default="./modelState/dm_candidROI_best_model.h5")
    parser.add_argument("--final-model", "-fm", dest="final_model", type=str, 
                        default="NOSAVE")

    args = parser.parse_args()
    run_opts = dict(
        img_extension=args.img_extension, 
        img_height=args.img_height,
        img_scale=args.img_scale,
        do_featurewise_norm=args.do_featurewise_norm,
        norm_fit_size=args.norm_fit_size,
        img_per_batch=args.img_per_batch,
        roi_per_img=args.roi_per_img,
        roi_size=tuple(args.roi_size),
        one_patch_mode=args.one_patch_mode,
        low_int_threshold=args.low_int_threshold,
        blob_min_area=args.blob_min_area,
        blob_min_int=args.blob_min_int,
        blob_max_int=args.blob_max_int,
        blob_th_step=args.blob_th_step,
        more_augmentation=args.more_augmentation,
        roi_state=args.roi_state,
        clf_bs=args.clf_bs,
        cutpoint=args.cutpoint,
        amp_factor=args.amp_factor,
        return_sample_weight=args.return_sample_weight,
        auto_batch_balance=args.auto_batch_balance,
        patches_per_epoch=args.patches_per_epoch, 
        nb_epoch=args.nb_epoch, 
        train_neg_vs_pos_ratio=args.train_neg_vs_pos_ratio,
        all_neg_skip=args.all_neg_skip,
        nb_init_filter=args.nb_init_filter, 
        init_filter_size=args.init_filter_size, 
        init_conv_stride=args.init_conv_stride, 
        pool_size=args.pool_size, 
        pool_stride=args.pool_stride, 
        weight_decay=args.weight_decay,
        alpha=args.alpha,
        l1_ratio=args.l1_ratio,
        inp_dropout=args.inp_dropout,
        hidden_dropout=args.hidden_dropout,
        init_lr=args.init_lr,
        val_size=args.val_size if args.val_size < 1 else int(args.val_size), 
        val_neg_vs_pos_ratio=args.val_neg_vs_pos_ratio,
        lr_patience=args.lr_patience, 
        es_patience=args.es_patience,
        resume_from=args.resume_from,
        net=args.net,
        load_val_ram=args.load_val_ram,
        load_train_ram=args.load_train_ram,
        no_pos_skip=args.no_pos_skip,
        exam_tsv=args.exam_tsv,
        img_tsv=args.img_tsv,
        best_model=args.best_model,        
        final_model=args.final_model        
    )
    print "\n>>> Model training options: <<<\n", run_opts, "\n"
    run(args.img_folder, **run_opts)



