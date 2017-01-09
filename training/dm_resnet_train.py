from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import keras.backend as K
from keras.callbacks import (
    ReduceLROnPlateau, 
    EarlyStopping, 
    # ModelCheckpoint
)
from keras.optimizers import SGD
from keras.callbacks import Callback
import os, argparse
import numpy as np
from meta import DMMetaManager
from dm_image import DMImageDataGenerator
from dm_resnet import (
    ResNetBuilder,
    MultiViewResNetBuilder
)
from dm_multi_gpu import make_parallel


class DMMetrics(object):
    '''Classification metrics for the DM challenge
    '''

    @staticmethod
    def sensitivity(y_true, y_pred):
        y_pred_pos = K.round(K.clip(y_pred, 0, 1))
        y_pos = K.round(K.clip(y_true, 0, 1))
        tp = K.sum(y_pos * y_pred_pos)
        pos = K.sum(y_pos)

        return tp / (pos + K.epsilon())

    @staticmethod
    def specificity(y_true, y_pred):
        y_pred_neg = 1 - K.round(K.clip(y_pred, 0, 1))
        y_neg = 1 - K.round(K.clip(y_true, 0, 1))
        tn = K.sum(y_neg * y_pred_neg)
        neg = K.sum(y_neg)

        return tn / (neg + K.epsilon())


class DMAucModelCheckpoint(Callback):
    '''Model checkpointer using AUROC score
    '''

    def __init__(self, filepath, test_data_gen, nb_test_samples):
        super(DMAucModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.test_data_gen = test_data_gen
        self.nb_test_samples = nb_test_samples
        self.best_epoch = None
        self.best_auc = 0.

    def on_epoch_end(self, epoch, logs={}):
        self.test_data_gen.reset()
        samples_seen = 0
        y_list = []
        pred_list = []
        while samples_seen < self.nb_test_samples:
            X, y = next(self.test_data_gen)
            samples_seen += len(y)
            y_list.append(y)
            pred_list.append(self.model.predict_on_batch(X))
        y_true = np.concatenate(y_list)
        y_pred = np.concatenate(pred_list)
        auc = roc_auc_score(y_true, y_pred)
        print " - Epoch:%d, AUROC: %.4f" % (epoch + 1, auc)
        if auc > self.best_auc:
            self.best_epoch = epoch + 1
            self.best_auc = auc
            self.model.save(self.filepath)

    def on_train_end(self, logs={}):
        print ">>> Found best AUROC: %.4f at epoch: %d, saved to: %s <<<" % \
            (self.best_auc, self.best_epoch, self.filepath)


def run(img_folder, img_extension='png', img_size=[288, 224], multi_view=False,
        do_featurewise_norm=True, featurewise_mean=7772., featurewise_std=12187., 
        batch_size=16, samples_per_epoch=160, nb_epoch=20, 
        balance_classes=.0, all_neg_skip=False, pos_cls_weight=1.0,
        nb_init_filter=64, init_filter_size=7, init_conv_stride=2, 
        pool_size=3, pool_stride=2, weight_decay=.0001, alpha=1., l1_ratio=.5, 
        inp_dropout=.0, hidden_dropout=.0, init_lr=.01,
        val_size=.2, lr_patience=5, es_patience=10, net='resnet50',
        exam_tsv='./metadata/exams_metadata.tsv',
        img_tsv='./metadata/images_crosswalk.tsv',
        best_model='./modelState/dm_resnet_best_model.h5',
        final_model='./modelState/dm_resnet_final_model.h5'):
    '''Run ResNet training on mammograms using an exam or image list
    Args:
        featurewise_mean, featurewise_std ([float]): they are estimated from 
                1152 x 896 pngs. Using different sized pngs give very close
                results.
    '''

    # Read some env variables.
    random_seed = int(os.getenv('RANDOM_SEED', 12345))
    nb_worker = int(os.getenv('NUM_CPU_CORES', 4))
    gpu_count = int(os.getenv('NUM_GPU_DEVICES', 1))
    
    # Setup training and validation data.
    meta_man = DMMetaManager(exam_tsv=exam_tsv, img_tsv=img_tsv, 
                             img_folder=img_folder, img_extension=img_extension)
    if multi_view:
        exam_list = meta_man.get_flatten_exam_list()
        exam_train, exam_val = train_test_split(
            exam_list, test_size=val_size, random_state=random_seed, 
            stratify=meta_man.exam_labs(exam_list))
        val_size_ = len(exam_val)*2  # L and R.
    else:
        img_list, lab_list = meta_man.get_flatten_img_list()
        img_train, img_val, lab_train, lab_val = train_test_split(
            img_list, lab_list, test_size=val_size, random_state=random_seed, 
            stratify=lab_list)
        val_size_ = len(img_val)

    train_img_gen = DMImageDataGenerator(
        horizontal_flip=True, 
        vertical_flip=True)
    val_img_gen = DMImageDataGenerator()
    if do_featurewise_norm:
        train_img_gen.featurewise_center = True
        train_img_gen.featurewise_std_normalization = True
        val_img_gen.featurewise_center = True
        val_img_gen.featurewise_std_normalization = True
        train_img_gen.mean = featurewise_mean
        train_img_gen.std = featurewise_std
        val_img_gen.mean = featurewise_mean
        val_img_gen.std = featurewise_std
    else:
        train_img_gen.samplewise_center = True
        train_img_gen.samplewise_std_normalization = True
        val_img_gen.samplewise_center = True
        val_img_gen.samplewise_std_normalization = True

    if multi_view:
        train_generator = train_img_gen.flow_from_exam_list(
            exam_train, target_size=(img_size[0], img_size[1]), 
            batch_size=batch_size, balance_classes=balance_classes, 
            all_neg_skip=all_neg_skip, shuffle=True, seed=random_seed,
            class_mode='binary')
        val_generator = val_img_gen.flow_from_exam_list(
            exam_val, target_size=(img_size[0], img_size[1]), 
            batch_size=batch_size, balance_classes=False, 
            all_neg_skip=False, shuffle=False, seed=random_seed, 
            class_mode='binary')
    else:
        train_generator = train_img_gen.flow_from_img_list(
            img_train, lab_train, target_size=(img_size[0], img_size[1]), 
            batch_size=batch_size, balance_classes=balance_classes, 
            all_neg_skip=all_neg_skip, shuffle=True, seed=random_seed,
            class_mode='binary')
        val_generator = val_img_gen.flow_from_img_list(
            img_val, lab_val, target_size=(img_size[0], img_size[1]), 
            batch_size=batch_size, balance_classes=False, 
            all_neg_skip=False, shuffle=False, seed=random_seed,
            class_mode='binary')

    # Create model.
    if multi_view:
        builder = MultiViewResNetBuilder
    else:
        builder = ResNetBuilder
    if net == 'resnet18':
        model = builder.build_resnet_18(
            (1, img_size[0], img_size[1]), 1, nb_init_filter, init_filter_size, 
            init_conv_stride, pool_size, pool_stride, weight_decay, alpha, l1_ratio, 
            inp_dropout, hidden_dropout)
    elif net == 'resnet34':
        model = builder.build_resnet_34(
            (1, img_size[0], img_size[1]), 1, nb_init_filter, init_filter_size, 
            init_conv_stride, pool_size, pool_stride, weight_decay, alpha, l1_ratio, 
            inp_dropout, hidden_dropout)
    elif net == 'resnet50':
        model = builder.build_resnet_50(
            (1, img_size[0], img_size[1]), 1, nb_init_filter, init_filter_size, 
            init_conv_stride, pool_size, pool_stride, weight_decay, alpha, l1_ratio, 
            inp_dropout, hidden_dropout)
    elif net == 'dmresnet14':
        model = builder.build_dm_resnet_14(
            (1, img_size[0], img_size[1]), 1, nb_init_filter, init_filter_size, 
            init_conv_stride, pool_size, pool_stride, weight_decay, alpha, l1_ratio, 
            inp_dropout, hidden_dropout)
    elif net == 'dmresnet47rb5':
        model = builder.build_dm_resnet_47rb5(
            (1, img_size[0], img_size[1]), 1, nb_init_filter, init_filter_size, 
            init_conv_stride, pool_size, pool_stride, weight_decay, alpha, l1_ratio, 
            inp_dropout, hidden_dropout)
    elif net == 'dmresnet56rb6':
        model = builder.build_dm_resnet_56rb6(
            (1, img_size[0], img_size[1]), 1, nb_init_filter, init_filter_size, 
            init_conv_stride, pool_size, pool_stride, weight_decay, alpha, l1_ratio, 
            inp_dropout, hidden_dropout)
    elif net == 'dmresnet65rb7':
        model = builder.build_dm_resnet_65rb7(
            (1, img_size[0], img_size[1]), 1, nb_init_filter, init_filter_size, 
            init_conv_stride, pool_size, pool_stride, weight_decay, alpha, l1_ratio, 
            inp_dropout, hidden_dropout)
    elif net == 'resnet101':
        model = builder.build_resnet_101(
            (1, img_size[0], img_size[1]), 1, nb_init_filter, init_filter_size, 
            init_conv_stride, pool_size, pool_stride, weight_decay, alpha, l1_ratio, 
            inp_dropout, hidden_dropout)
    elif net == 'resnet152':
        model = builder.build_resnet_152(
            (1, img_size[0], img_size[1]), 1, nb_init_filter, init_filter_size, 
            init_conv_stride, pool_size, pool_stride, weight_decay, alpha, l1_ratio, 
            inp_dropout, hidden_dropout)
    
    if gpu_count > 1:
        model = make_parallel(model, gpu_count)

    # Model training.
    sgd = SGD(lr=init_lr, momentum=0.9, decay=0.0, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy', 
                  metrics=[DMMetrics.sensitivity, DMMetrics.specificity])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, 
                                  patience=lr_patience, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=es_patience, verbose=1)
    auc_checkpointer = DMAucModelCheckpoint(best_model, val_generator, val_size_)
    # checkpointer = ModelCheckpoint(
    #     best_model, monitor='val_loss', verbose=1, save_best_only=True)
    hist = model.fit_generator(
        train_generator, 
        samples_per_epoch=samples_per_epoch, 
        nb_epoch=nb_epoch,
        class_weight={ 0: 1.0, 1: pos_cls_weight },
        validation_data=val_generator, 
        nb_val_samples=val_size_, 
        callbacks=[reduce_lr, early_stopping, auc_checkpointer], 
        nb_worker=nb_worker, 
        pickle_safe=True,  # turn on pickle_safe to avoid a strange error.
        verbose=2
        )

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
    
    model.save(final_model)

    return hist


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="DM ResNet training")
    parser.add_argument("img_folder", type=str)
    parser.add_argument("--img-extension", "-ext", dest="img_extension", 
                        type=str, default="png")
    parser.add_argument("--img-size", "-is", dest="img_size", nargs=2, type=int, 
                        default=[288, 224])
    parser.add_argument("--multi-view", dest="multi_view", action="store_true")
    parser.add_argument("--no-multi-view", dest="multi_view", action="store_false")
    parser.set_defaults(multi_view=False)
    parser.add_argument("--featurewise-norm", dest="do_featurewise_norm", action="store_true")
    parser.add_argument("--no-featurewise-norm", dest="do_featurewise_norm", action="store_false")
    parser.set_defaults(do_featurewise_norm=True)
    parser.add_argument("--featurewise-mean", "-feam", dest="featurewise_mean", 
                        type=float, default=7772.)
    parser.add_argument("--featurewise-std", "-feas", dest="featurewise_std", 
                        type=float, default=12187.)
    parser.add_argument("--batch-size", "-bs", dest="batch_size", type=int, default=16)
    parser.add_argument("--samples-per-epoch", "-spe", dest="samples_per_epoch", 
                        type=int, default=160)
    parser.add_argument("--nb-epoch", "-ne", dest="nb_epoch", type=int, default=20)
    parser.add_argument("--balance-classes", "-bc", dest="balance_classes", type=float, default=.0)
    parser.add_argument("--allneg-skip", dest="all_neg_skip", action="store_true")
    parser.add_argument("--no-allneg-skip", dest="all_neg_skip", action="store_false")
    parser.set_defaults(all_neg_skip=False)
    parser.add_argument("--pos-class-weight", "-pcw", dest="pos_cls_weight", type=float, default=1.0)
    parser.add_argument("--nb-init-filter", "-nif", dest="nb_init_filter", type=int, default=64)
    parser.add_argument("--init-filter-size", "-ifs", dest="init_filter_size", type=int, default=7)
    parser.add_argument("--init-conv-stride", "-ics", dest="init_conv_stride", type=int, default=2)
    parser.add_argument("--max-pooling-size", "-mps", dest="pool_size", type=int, default=3)
    parser.add_argument("--max-pooling-stride", "-mpr", dest="pool_stride", type=int, default=2)
    parser.add_argument("--weight-decay", "-wd", dest="weight_decay", 
                        type=float, default=.0001)
    parser.add_argument("--alpha", dest="alpha", type=float, default=1.)
    parser.add_argument("--l1-ratio", dest="l1_ratio", type=float, default=.5)
    parser.add_argument("--inp-dropout", "-id", dest="inp_dropout", type=float, default=.0)
    parser.add_argument("--hidden-dropout", "-hd", dest="hidden_dropout", type=float, default=.0)
    parser.add_argument("--init-learningrate", "-ilr", dest="init_lr", type=float, default=.01)
    parser.add_argument("--val-size", "-vs", dest="val_size", type=float, default=.2)
    parser.add_argument("--lr-patience", "-lrp", dest="lr_patience", type=int, default=5)
    parser.add_argument("--es-patience", "-esp", dest="es_patience", type=int, default=10)
    parser.add_argument("--net", dest="net", type=str, default="resnet50")
    # parser.add_argument("--nb-worker", "-nw", dest="nb_worker", type=int, default=4)
    parser.add_argument("--exam-tsv", "-et", dest="exam_tsv", type=str, 
                        default="./metadata/exams_metadata.tsv")
    parser.add_argument("--img-tsv", "-it", dest="img_tsv", type=str, 
                        default="./metadata/images_crosswalk.tsv")
    parser.add_argument("--best-model", "-bm", dest="best_model", type=str, 
                        default="./modelState/dm_resnet_best_model.h5")
    parser.add_argument("--final-model", "-fm", dest="final_model", type=str, 
                        default="./modelState/dm_resnet_final_model.h5")

    args = parser.parse_args()
    run_opts = dict(
        img_extension=args.img_extension, 
        img_size=args.img_size, 
        multi_view=args.multi_view,
        do_featurewise_norm=args.do_featurewise_norm,
        featurewise_mean=args.featurewise_mean,
        featurewise_std=args.featurewise_std,
        batch_size=args.batch_size, 
        samples_per_epoch=args.samples_per_epoch, 
        nb_epoch=args.nb_epoch, 
        balance_classes=args.balance_classes,
        all_neg_skip=args.all_neg_skip,
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
        val_size=args.val_size if args.val_size < 1 else int(args.val_size), 
        lr_patience=args.lr_patience, 
        es_patience=args.es_patience,
        net=args.net,
        # nb_worker=args.nb_worker,
        exam_tsv=args.exam_tsv,
        img_tsv=args.img_tsv,
        best_model=args.best_model,        
        final_model=args.final_model        
    )
    print "\n>>> Model training options: <<<\n", run_opts, "\n"
    run(args.img_folder, **run_opts)


