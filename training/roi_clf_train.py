import os, argparse
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import (
    ReduceLROnPlateau, 
    EarlyStopping, 
)
from keras.models import load_model
from dm_keras_ext import DMMetrics, DMAucModelCheckpoint
from dm_resnet import ResNetBuilder

import warnings
import exceptions
warnings.filterwarnings('ignore', category=exceptions.UserWarning)


def resize_img_dat(img_dat, img_size):
    '''Resize a train or test image ndarray dataset
    '''
    import cv2

    if img_dat.shape[1:3] != tuple(img_size):
        resized_dat = np.zeros(
            (img_dat.shape[0],) + tuple(img_size) + (img_dat.shape[3],) )
        for i,img in enumerate(img_dat):
            img_ = cv2.resize(
                img, dsize=(img_size[1], img_size[0]), 
                interpolation=cv2.INTER_CUBIC)
            resized_dat[i] = img_.reshape(img_.shape + (img.shape[2],))
        return resized_dat
    else:
        return img_dat


def run(x_train_fn, x_test_fn, y_train_fn, y_test_fn, 
        img_size=[256, 256], do_featurewise_norm=True, 
        rotation_range=0, width_shift_range=.0, height_shift_range=.0,
        zoom_range=[1.0, 1.0], horizontal_flip=False, vertical_flip=False,
        batch_size=32, nb_epoch=100, pos_cls_weight=1.0,
        nb_init_filter=32, init_filter_size=5, init_conv_stride=2, 
        pool_size=2, pool_stride=2, 
        weight_decay=.0001, alpha=.0001, l1_ratio=.0, 
        inp_dropout=.0, hidden_dropout=.0, init_lr=.01,
        lr_patience=20, es_patience=40,
        resume_from=None, 
        best_model='./modelState/roi_clf.h5',
        final_model="NOSAVE"):
    '''Train a deep learning model for ROI classifications
    '''

    # =========== Load training data =============== #
    X_train = np.load(x_train_fn)
    X_test = np.load(x_test_fn)
    X_train = resize_img_dat(X_train, img_size)
    X_test = resize_img_dat(X_test, img_size)
    y_train = np.load(y_train_fn)
    y_test = np.load(y_test_fn)

    # ============ Train & validation set =============== #
    if do_featurewise_norm:
        imgen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True)
        imgen.fit(X_train)
    else:
        imgen = ImageDataGenerator(
            samplewise_center=True,
            samplewise_std_normalization=True)
    imgen.rotation_range = rotation_range
    imgen.width_shift_range = width_shift_range
    imgen.height_shift_range = height_shift_range
    imgen.zoom_range = zoom_range
    imgen.horizontal_flip = horizontal_flip
    imgen.vertical_flip = vertical_flip
    train_generator = imgen.flow(X_train, y_train, batch_size=batch_size, 
                                 shuffle=True, seed=12345)
    
    X_test -= imgen.mean
    X_test /= imgen.std
    validation_set = (X_test, y_test)

    # ================= Model training ============== #
    nb_worker = int(os.getenv('NUM_CPU_CORES', 4))
    if resume_from is not None:
        model = load_model(
            resume_from, 
            custom_objects={
                'sensitivity': DMMetrics.sensitivity, 
                'specificity': DMMetrics.specificity
            }
        )
    else:
        model = ResNetBuilder.build_resnet_50(
            (1, img_size[0], img_size[1]), 1, 
            nb_init_filter, init_filter_size, init_conv_stride, 
            pool_size, pool_stride, 
            weight_decay, alpha, l1_ratio, 
            inp_dropout, hidden_dropout)
    sgd = SGD(lr=init_lr, momentum=0.9, decay=0.0, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy', 
                  metrics=[DMMetrics.sensitivity, DMMetrics.specificity])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, 
                                  patience=lr_patience, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=es_patience, verbose=1)
    auc_checkpointer = DMAucModelCheckpoint(
        best_model, validation_set, batch_size=batch_size)
    hist = model.fit_generator(
        train_generator, 
        samples_per_epoch=len(X_train), 
        nb_epoch=nb_epoch,
        class_weight={ 0: 1.0, 1: pos_cls_weight },
        validation_data=validation_set, 
        callbacks=[reduce_lr, early_stopping, auc_checkpointer], 
        nb_worker=nb_worker, 
        pickle_safe=True,  # turn on pickle_safe to avoid a strange error.
        verbose=2)

    # Training report.
    min_loss_locs, = np.where(hist.history['val_loss'] == min(hist.history['val_loss']))
    best_val_loss = hist.history['val_loss'][min_loss_locs[0]]
    best_val_sensitivity = hist.history['val_sensitivity'][min_loss_locs[0]]
    best_val_specificity = hist.history['val_specificity'][min_loss_locs[0]]
    print "\n==== Training summary ===="
    print "Minimum val loss achieved at epoch:", min_loss_locs[0] + 1
    print "Best val loss:", best_val_loss
    print "Best val sensitivity:", best_val_sensitivity
    print "Best val specificity:", best_val_specificity

    if final_model != "NOSAVE":
        model.save(final_model)

    return hist


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="DM ROI clf training")
    parser.add_argument("x_train_fn", type=str)
    parser.add_argument("x_test_fn", type=str)
    parser.add_argument("y_train_fn", type=str)
    parser.add_argument("y_test_fn", type=str)

    parser.add_argument("--img-size", "-is", dest="img_size", nargs=2, type=int, 
                        default=[256, 256])
    parser.add_argument("--featurewise-norm", dest="do_featurewise_norm", action="store_true")
    parser.add_argument("--no-featurewise-norm", dest="do_featurewise_norm", action="store_false")
    parser.set_defaults(do_featurewise_norm=True)
    parser.add_argument("--batch-size", "-bs", dest="batch_size", type=int, default=32)
    parser.add_argument("--rotation-range", dest="rotation_range", type=int, default=0)
    parser.add_argument("--width-shift-range", dest="width_shift_range", type=float, default=.0)
    parser.add_argument("--height-shift-range", dest="height_shift_range", type=float, default=.0)
    parser.add_argument("--zoom-range", dest="zoom_range", nargs=2, type=float, default=[1.0, 1.0])
    parser.add_argument("--horizontal-flip", dest="horizontal_flip", action="store_true")
    parser.add_argument("--no-horizontal-flip", dest="horizontal_flip", action="store_false")
    parser.set_defaults(horizontal_flip=False)
    parser.add_argument("--vertical-flip", dest="vertical_flip", action="store_true")
    parser.add_argument("--no-vertical-flip", dest="vertical_flip", action="store_false")
    parser.set_defaults(vertical_flip=False)
    parser.add_argument("--nb-epoch", "-ne", dest="nb_epoch", type=int, default=100)
    parser.add_argument("--pos-class-weight", "-pcw", dest="pos_cls_weight", type=float, default=1.0)
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
    parser.add_argument("--lr-patience", "-lrp", dest="lr_patience", type=int, default=20)
    parser.add_argument("--es-patience", "-esp", dest="es_patience", type=int, default=40)
    parser.add_argument("--resume-from", dest="resume_from", type=str, default=None)
    parser.add_argument("--no-resume-from", dest="resume_from", action="store_const", const=None)
    parser.add_argument("--best-model", "-bm", dest="best_model", type=str, 
                        default="./modelState/roi_clf.h5")
    parser.add_argument("--final-model", "-fm", dest="final_model", type=str, 
                        default="NOSAVE")

    args = parser.parse_args()
    run_opts = dict(
        img_size=args.img_size, 
        do_featurewise_norm=args.do_featurewise_norm,
        batch_size=args.batch_size, 
        rotation_range=args.rotation_range,
        width_shift_range=args.width_shift_range,
        height_shift_range=args.height_shift_range,
        zoom_range=args.zoom_range,
        horizontal_flip=args.horizontal_flip,
        vertical_flip=args.vertical_flip,
        nb_epoch=args.nb_epoch, 
        pos_cls_weight=args.pos_cls_weight,
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
        lr_patience=args.lr_patience, 
        es_patience=args.es_patience,
        resume_from=args.resume_from,
        best_model=args.best_model,        
        final_model=args.final_model        
    )
    print "\n>>> Model training options: <<<\n", run_opts, "\n"
    run(args.x_train_fn, args.x_test_fn, args.y_train_fn, args.y_test_fn, 
        **run_opts)









