import os, argparse
import pickle
import numpy as np
from numpy.random import RandomState
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    roc_auc_score, 
    precision_score, 
    recall_score,
    log_loss
)
from sklearn.model_selection import train_test_split
# from sklearn.exceptions import UndefinedMetricWarning
from keras.models import load_model, Model
from meta import DMMetaManager
from dm_image import DMImageDataGenerator
from dm_multi_gpu import make_parallel
from dm_keras_ext import DMMetrics

import warnings
import exceptions
warnings.filterwarnings('ignore', category=exceptions.UserWarning)


def run(img_folder, img_extension='png', img_size=[288, 224], multi_view=False,
        do_featurewise_norm=True, featurewise_mean=7772., featurewise_std=12187., 
        batch_size=16, samples_per_epoch=160, nb_epoch=20, 
        balance_classes=0., all_neg_skip=False, pos_cls_weight=1.0,
        alpha=1., l1_ratio=.5, init_lr=.01, power_t=.25, val_size=.2, 
        exam_tsv='./metadata/exams_metadata.tsv',
        img_tsv='./metadata/images_crosswalk.tsv',
        dl_state='./modelState/resnet50_288_best_model.h5',
        best_model='./modelState/enet_288_best_model.h5',
        final_model="NOSAVE"):

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

    img_gen = DMImageDataGenerator(
        horizontal_flip=True, 
        vertical_flip=True)
    if do_featurewise_norm:
        img_gen.featurewise_center = True
        img_gen.featurewise_std_normalization = True
        img_gen.mean = featurewise_mean
        img_gen.std = featurewise_std
    else:
        img_gen.samplewise_center = True
        img_gen.samplewise_std_normalization = True

    if multi_view:
        train_generator = img_gen.flow_from_exam_list(
            exam_train, target_size=(img_size[0], img_size[1]), 
            batch_size=batch_size, balance_classes=balance_classes, 
            all_neg_skip=all_neg_skip, shuffle=True, seed=random_seed,
            class_mode='binary')
        val_generator = img_gen.flow_from_exam_list(
            exam_val, target_size=(img_size[0], img_size[1]), 
            batch_size=batch_size, validation_mode=True, 
            class_mode='binary')
    else:
        train_generator = img_gen.flow_from_img_list(
            img_train, lab_train, target_size=(img_size[0], img_size[1]), 
            batch_size=batch_size, balance_classes=balance_classes, 
            all_neg_skip=all_neg_skip, shuffle=True, seed=random_seed,
            class_mode='binary')
        val_generator = img_gen.flow_from_img_list(
            img_val, lab_val, target_size=(img_size[0], img_size[1]), 
            batch_size=batch_size, validation_mode=True,
            class_mode='binary')


    # Deep learning model.
    dl_model = load_model(
            dl_state, 
            custom_objects={'sensitivity': DMMetrics.sensitivity, 
                            'specificity': DMMetrics.specificity})
    # Dummy compilation to turn off the "uncompiled" error when model was run on multi-GPUs.
    # dl_model.compile(optimizer='sgd', loss='binary_crossentropy')
    reprlayer_model = Model(
        input=dl_model.input, output=dl_model.get_layer(index=-2).output)
    if gpu_count > 1:
        reprlayer_model = make_parallel(reprlayer_model, gpu_count)


    # Setup test data in RAM.
    X_list = []
    y_list = []
    samples_seen = 0
    while samples_seen < val_size_:
        X, y = next(val_generator)
        X_list.append(reprlayer_model.predict_on_batch(X))
        y_list.append(y)
        samples_seen += len(y)
    X_test = np.concatenate(X_list)
    y_test = np.concatenate(y_list)
    del X_list
    del y_list


    # Evaluat DL model on the test data.
    val_generator.reset()
    dl_test_pred = dl_model.predict_generator(
            val_generator, val_samples=val_size_, nb_worker=nb_worker)
    dl_auc = roc_auc_score(y_test, dl_test_pred)
    print "\nAUROC by the DL model:", dl_auc

    # Elastic net training.
    target_classes = np.array([0, 1])
    sgd_clf = SGDClassifier(
        loss='log', penalty='elasticnet', alpha=alpha, l1_ratio=l1_ratio, 
        verbose=0, n_jobs=nb_worker, learning_rate='invscaling', eta0=init_lr, 
        power_t=power_t)

    best_epoch = 0
    best_auc = 0.
    for epoch in xrange(nb_epoch):
        samples_seen = 0
        # train_generator.reset()
        while samples_seen < samples_per_epoch:
            X, y = next(train_generator)
            X_repr = reprlayer_model.predict_on_batch(X)
            sgd_clf.partial_fit(X_repr, y, classes=target_classes)
            samples_seen += len(y)
        # End of epoch summary.
        pred_prob = sgd_clf.predict_proba(X_test)[:, 1]
        # pred_lab = np.array(pred_prob > .5, dtype='int32')
        auc = roc_auc_score(y_test, pred_prob)
        if auc > best_auc:
            best_epoch = epoch + 1
            best_auc = auc
            if best_model != "NOSAVE":
                with open(best_model, 'w') as best_state:
                    pickle.dump(sgd_clf, best_state)
        crossentropy_loss = log_loss(y_test, pred_prob)
        # precision = precision_score(y_test, pred_lab)
        # recall = recall_score(y_test, pred_lab)
        wei_sparseness = np.mean(sgd_clf.coef_ == 0)
        print ("Epoch=%d, auc=%.4f, loss=%.4f, weight sparsity=%.4f") % \
            (epoch + 1, auc, crossentropy_loss, wei_sparseness)
    # End of training summary
    print ">>> Found best AUROC: %.4f at epoch: %d, saved to: %s <<<" % \
        (best_auc, best_epoch, best_model)

    #### Save elastic net model!! ####
    if final_model != "NOSAVE":
        with open(final_model, 'w') as final_state:
            pickle.dump(sgd_clf, final_state)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="DM ElasticNet training")
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
    parser.add_argument("--allneg-skip", dest="all_neg_skip", type=float, default=0.)
    parser.add_argument("--pos-class-weight", "-pcw", dest="pos_cls_weight", type=float, default=1.0)
    parser.add_argument("--alpha", dest="alpha", type=float, default=1.)
    parser.add_argument("--l1-ratio", dest="l1_ratio", type=float, default=.5)
    parser.add_argument("--init-learningrate", "-ilr", dest="init_lr", type=float, default=.01)
    parser.add_argument("--power-t", "-pt", dest="power_t", type=float, default=.25)
    parser.add_argument("--val-size", "-vs", dest="val_size", type=float, default=.2)
    # parser.add_argument("--resume-from", "-rf", dest="resume_from", type=str, default=None)
    parser.add_argument("--exam-tsv", "-et", dest="exam_tsv", type=str, 
                        default="./metadata/exams_metadata.tsv")
    parser.add_argument("--img-tsv", "-it", dest="img_tsv", type=str, 
                        default="./metadata/images_crosswalk.tsv")
    parser.add_argument("--dl-state", "-ds", dest="dl_state", type=str, default="none")
    parser.add_argument("--best-model", "-bm", dest="best_model", type=str, 
                        default="./modelState/enet_288_best_model.h5")
    parser.add_argument("--final-model", "-fm", dest="final_model", type=str, 
                        default="NOSAVE")

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
        alpha=args.alpha,
        l1_ratio=args.l1_ratio,
        init_lr=args.init_lr,
        power_t=args.power_t,
        val_size=args.val_size if args.val_size < 1 else int(args.val_size), 
        # resume_from=args.resume_from,
        exam_tsv=args.exam_tsv,
        img_tsv=args.img_tsv,
        dl_state=args.dl_state,
        best_model=args.best_model,        
        final_model=args.final_model        
    )
    print "\n>>> Model training options: <<<\n", run_opts, "\n"
    run(args.img_folder, **run_opts)
