import argparse
import os, sys
import pickle
import numpy as np
from numpy.random import RandomState
from scipy.misc import toimage
from sklearn.model_selection import train_test_split
from keras.models import load_model
from meta import DMMetaManager
from dm_image import add_img_margins, read_resize_img, sweep_img_patches
from dm_image import DMImageDataGenerator
from dm_keras_ext import (
    DMMetrics as dmm, 
    get_dl_model, create_optimizer, 
    load_dat_ram,
    do_3stage_training
)
from dm_multi_gpu import make_parallel
from dm_preprocess import DMImagePreprocessor as prep

import warnings
import exceptions
warnings.filterwarnings('ignore', category=exceptions.UserWarning)
import keras.backend as K
dim_ordering = K.image_dim_ordering()


def score_write_patches(img_list, lab_list, target_height, target_scale, 
                        patch_size, stride, model, batch_size, preprocess, 
                        neg_out, pos_out, bkg_out, roi_cutoff=.9, bkg_cutoff=.5, 
                        sample_bkg=True, img_ext='png', random_seed=12345):
    '''Score image patches and write them to an external directory
    '''
    def write_patches(img_fn, patch_dat, idx, out_dir, img_ext='png'):
        basename = os.path.basename(img_fn)
        fn_no_ext = os.path.splitext(basename)[0]
        if img_ext == 'png':
            max_val = 65535.
        else:
            max_val = 255.
        for i in idx:
            # import pdb; pdb.set_trace()
            patch = patch_dat[i]
            patch_max = patch.max() if patch.max() != 0 else max_val
            patch *= max_val/patch_max
            patch = patch.astype('int32')
            mode = 'I' if img_ext == 'png' else None
            patch_img = toimage(patch, high=patch.max(), low=patch.min(),
                                mode=mode)
            filename = fn_no_ext + "_%06d" % (i) + '.' + img_ext
            fullname = os.path.join(out_dir, filename)
            # import pdb; pdb.set_trace()
            patch_img.save(fullname)


    rng = RandomState(random_seed)
    nb_roi = 0
    nb_bkg = 0
    for img_fn, img_lab in zip(img_list, lab_list):
        img = read_resize_img(img_fn, target_height=target_height)
        img,_ = prep.segment_breast(img)
        img = add_img_margins(img, patch_size/2)
        # import pdb; pdb.set_trace()
        patch_dat, nb_row, nb_col = sweep_img_patches(
            img, patch_size, stride, target_scale=target_scale)
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
        roi_idx = np.where(pred[:,0] < 1 - roi_cutoff)[0]
        bkg_idx = np.where(pred[:,0] > bkg_cutoff)[0]
        if sample_bkg and len(bkg_idx) > len(roi_idx):
            bkg_idx = rng.choice(bkg_idx, len(roi_idx), replace=False)
        roi_out = pos_out if img_lab==1 else neg_out
        write_patches(img_fn, patch_dat, roi_idx, roi_out, img_ext)
        write_patches(img_fn, patch_dat, bkg_idx, bkg_out, img_ext)
        nb_roi += len(roi_idx)
        nb_bkg += len(bkg_idx)
    return nb_roi, nb_bkg
        

def run(img_folder, dl_state, best_model, img_extension='dcm', 
        img_height=1024, img_scale=255., neg_vs_pos_ratio=1., 
        val_size=.1, test_size=.15,
        net='vgg19', batch_size=128, patch_size=256, stride=8,
        roi_cutoff=.9, bkg_cutoff=.5, sample_bkg=True,
        train_out='./scratch/train', val_out='./scratch/val', 
        test_out='./scratch/test', out_img_ext='png',
        neg_name='benign', pos_name='malignant', bkg_name='background',
        augmentation=True, load_train_ram=False, load_val_ram=False,
        top_layer_nb=None, nb_epoch=10, top_layer_epochs=0, all_layer_epochs=0,
        optim='sgd', init_lr=.01, 
        top_layer_multiplier=.01, all_layer_multiplier=.0001,
        es_patience=5, lr_patience=2, weight_decay2=.01, bias_multiplier=.1,
        hidden_dropout2=.0,
        exam_tsv='./metadata/exams_metadata.tsv',
        img_tsv='./metadata/images_crosswalk.tsv',
        out='./modelState/subj_lists.pkl'):
    '''Finetune a trained DL model on a different dataset
    '''
    # Read some env variables.
    random_seed = int(os.getenv('RANDOM_SEED', 12345))
    rng = RandomState(random_seed)  # an rng used across board.
    nb_worker = int(os.getenv('NUM_CPU_CORES', 4))    
    gpu_count = int(os.getenv('NUM_GPU_DEVICES', 1))

    # Load and split image and label lists.
    meta_man = DMMetaManager(exam_tsv=exam_tsv, 
                             img_tsv=img_tsv, 
                             img_folder=img_folder, 
                             img_extension=img_extension)
    subj_list, subj_labs = meta_man.get_subj_labs()
    subj_labs = np.array(subj_labs)
    print "Found %d subjests" % (len(subj_list))
    print "cancer patients=%d, normal patients=%d" \
            % ((subj_labs==1).sum(), (subj_labs==0).sum())
    if neg_vs_pos_ratio is not None:
        subj_list, subj_labs = DMMetaManager.subset_subj_list(
            subj_list, subj_labs, neg_vs_pos_ratio, random_seed)
        subj_labs = np.array(subj_labs)
        print "After subsetting, there are %d subjects" % (len(subj_list))
        print "cancer patients=%d, normal patients=%d" \
                % ((subj_labs==1).sum(), (subj_labs==0).sum())
    subj_train, subj_test, labs_train, labs_test = train_test_split(
        subj_list, subj_labs, test_size=test_size, stratify=subj_labs, 
        random_state=random_seed)
    subj_train, subj_val, labs_train, labs_val = train_test_split(
        subj_train, labs_train, test_size=val_size, stratify=labs_train, 
        random_state=random_seed)

    # Get image lists.
    # >>>> Debug <<<< #
    # subj_train = subj_train[:1]
    # subj_val = subj_val[:1]
    # subj_test = subj_test[:1]
    # >>>> Debug <<<< #
    print "Get flattened image lists"
    img_train, ilab_train = meta_man.get_flatten_img_list(subj_train)
    img_val, ilab_val = meta_man.get_flatten_img_list(subj_val)
    img_test, ilab_test = meta_man.get_flatten_img_list(subj_test)
    ilab_train = np.array(ilab_train)
    ilab_val = np.array(ilab_val)
    ilab_test = np.array(ilab_test)
    print "On train set, positive img=%d, negative img=%d" \
            % ((ilab_train==1).sum(), (ilab_train==0).sum())
    print "On val set, positive img=%d, negative img=%d" \
            % ((ilab_val==1).sum(), (ilab_val==0).sum())
    print "On test set, positive img=%d, negative img=%d" \
            % ((ilab_test==1).sum(), (ilab_test==0).sum())
    sys.stdout.flush()

    # Load DL model, preprocess function.
    print "Load patch classifier:", dl_state; sys.stdout.flush()
    dl_model, preprocess_input, top_layer_nb = get_dl_model(
        net, use_pretrained=True, resume_from=dl_state,
        top_layer_nb=top_layer_nb)    
    if gpu_count > 1:
        print "Make the model parallel on %d GPUs" % (gpu_count)
        sys.stdout.flush()
        dl_model = make_parallel(dl_model, gpu_count)

    # Sweep the whole images and classify patches.
    print "Score image patches and write them to:", train_out
    sys.stdout.flush()
    nb_roi_train, nb_bkg_train = score_write_patches(
        img_train, ilab_train, img_height, img_scale,
        patch_size, stride, dl_model, batch_size, 
        preprocess_input, 
        neg_out=os.path.join(train_out, neg_name),
        pos_out=os.path.join(train_out, pos_name),
        bkg_out=os.path.join(train_out, bkg_name),
        roi_cutoff=roi_cutoff, bkg_cutoff=bkg_cutoff,
        sample_bkg=sample_bkg, img_ext=out_img_ext, random_seed=random_seed)
    print "Wrote %d ROI and %d bkg patches" % (nb_roi_train, nb_bkg_train)
    ####
    print "Score image patches and write them to:", val_out
    sys.stdout.flush()
    nb_roi_val, nb_bkg_val = score_write_patches(
        img_val, ilab_val, img_height, img_scale,
        patch_size, stride, dl_model, batch_size, 
        preprocess_input, 
        neg_out=os.path.join(val_out, neg_name),
        pos_out=os.path.join(val_out, pos_name),
        bkg_out=os.path.join(val_out, bkg_name),
        roi_cutoff=roi_cutoff, bkg_cutoff=bkg_cutoff,
        sample_bkg=sample_bkg, img_ext=out_img_ext, random_seed=random_seed)
    print "Wrote %d ROI and %d bkg patches" % (nb_roi_val, nb_bkg_val)
    ####
    print "Score image patches and write them to:", test_out
    sys.stdout.flush()
    nb_roi_test, nb_bkg_test = score_write_patches(
        img_test, ilab_test, img_height, img_scale,
        patch_size, stride, dl_model, batch_size, 
        preprocess_input, 
        neg_out=os.path.join(test_out, neg_name),
        pos_out=os.path.join(test_out, pos_name),
        bkg_out=os.path.join(test_out, bkg_name),
        roi_cutoff=roi_cutoff, bkg_cutoff=bkg_cutoff,
        sample_bkg=sample_bkg, img_ext=out_img_ext, random_seed=random_seed)
    print "Wrote %d ROI and %d bkg patches" % (nb_roi_test, nb_bkg_test)
    sys.stdout.flush()

    # ==== Image generators ==== #
    train_imgen = DMImageDataGenerator()
    val_imgen = DMImageDataGenerator()
    if augmentation:
        train_imgen.horizontal_flip=True 
        train_imgen.vertical_flip=True
        train_imgen.rotation_range = 45.
        train_imgen.shear_range = np.pi/8.

    # ==== Train & val set ==== #
    if load_train_ram:
        raw_imgen = DMImageDataGenerator()
        print "Create generator for raw train set"
        raw_generator = raw_imgen.flow_from_directory(
            train_out, target_size=(patch_size, patch_size), 
            target_scale=img_scale, dup_3_channels=True,
            classes=[bkg_name, pos_name, neg_name], class_mode='categorical', 
            batch_size=batch_size, shuffle=False)
        print "Loading raw train set into RAM.",
        sys.stdout.flush()
        raw_set = load_dat_ram(raw_generator, raw_generator.nb_sample)
        print "Done."; sys.stdout.flush()
        print "Create generator for train set"
        train_generator = train_imgen.flow(
            raw_set[0], raw_set[1], batch_size=batch_size, 
            auto_batch_balance=True, preprocess=preprocess_input, 
            shuffle=True, seed=random_seed)
    else:
        print "Create generator for train set"
        train_generator = train_imgen.flow_from_directory(
            train_out, target_size=(patch_size, patch_size), 
            target_scale=img_scale, dup_3_channels=True,
            classes=[bkg_name, pos_name, neg_name], class_mode='categorical', 
            auto_batch_balance=True, batch_size=batch_size, 
            preprocess=preprocess_input, shuffle=True, seed=random_seed)

    print "Create generator for val set"
    validation_set = val_imgen.flow_from_directory(
        val_out, target_size=(patch_size, patch_size), target_scale=img_scale,
        dup_3_channels=True, classes=[bkg_name, pos_name, neg_name], 
        class_mode='categorical', batch_size=batch_size, 
        preprocess=preprocess_input, shuffle=False)
    sys.stdout.flush()
    if load_val_ram:
        print "Loading validation set into RAM.",
        sys.stdout.flush()
        validation_set = load_dat_ram(validation_set, validation_set.nb_sample)
        print "Done."; sys.stdout.flush()

    # ==== Model finetuning ==== #
    # import pdb; pdb.set_trace()
    dl_model, loss_hist, acc_hist = do_3stage_training(
        dl_model, train_generator, validation_set, best_model, 
        train_generator.nb_sample, top_layer_nb, net, nb_epoch=nb_epoch,
        top_layer_epochs=top_layer_epochs, all_layer_epochs=all_layer_epochs,
        use_pretrained=True, optim=optim, init_lr=init_lr, 
        top_layer_multiplier=top_layer_multiplier, 
        all_layer_multiplier=all_layer_multiplier,
        es_patience=es_patience, lr_patience=lr_patience, 
        auto_batch_balance=True, nb_worker=nb_worker,
        weight_decay2=weight_decay2, bias_multiplier=bias_multiplier,
        hidden_dropout2=hidden_dropout2)

    # Training report.
    min_loss_locs, = np.where(loss_hist == min(loss_hist))
    best_val_loss = loss_hist[min_loss_locs[0]]
    best_val_accuracy = acc_hist[min_loss_locs[0]]
    print "\n==== Training summary ===="
    print "Minimum val loss achieved at epoch:", min_loss_locs[0] + 1
    print "Best val loss:", best_val_loss
    print "Best val accuracy:", best_val_accuracy

    # ==== Predict on test set ==== #
    print "\n==== Predicting on test set ===="
    test_imgen = DMImageDataGenerator()
    print "Create generator for test set"
    test_generator = test_imgen.flow_from_directory(
        test_out, target_size=(patch_size, patch_size), target_scale=img_scale,
        dup_3_channels=True, classes=[bkg_name, pos_name, neg_name], 
        class_mode='categorical', batch_size=batch_size, 
        preprocess=preprocess_input, shuffle=False)
    print "Load saved best model:", best_model + '.',
    sys.stdout.flush()
    saved_model = load_model(best_model)
    print "Done."
    test_res = saved_model.evaluate_generator(
        test_generator, test_generator.nb_sample, nb_worker=nb_worker, 
        pickle_safe=True)
    print "Evaluation result on test set:", test_res

    # Save the result.
    print "Saving subject lists to external files.",
    sys.stdout.flush()
    pickle.dump((subj_train, subj_val, subj_test), open(out, 'w'))
    print "Done."

 
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="DM DL model finetuning")
    parser.add_argument("img_folder", type=str)
    parser.add_argument("dl_state", type=str)
    parser.add_argument("best_model", type=str)
    parser.add_argument("--img-extension", "-ext", dest="img_extension", type=str, default="dcm")
    parser.add_argument("--img-height", "-ih", dest="img_height", type=int, default=1024)
    parser.add_argument("--img-scale", "-ic", dest="img_scale", type=float, default=255.)
    parser.add_argument("--neg-vs-pos-ratio", dest="neg_vs_pos_ratio", type=float, default=10.)
    parser.add_argument("--no-neg-vs-pos-ratio", dest="neg_vs_pos_ratio", 
                        action="store_const", const=None)
    parser.add_argument("--test-size", dest="test_size", type=float, default=.15)
    parser.add_argument("--val-size", dest="val_size", type=float, default=.1)
    parser.add_argument("--net", dest="net", type=str, default="vgg19")    
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=128)
    parser.add_argument("--patch-size", dest="patch_size", type=int, default=256)
    parser.add_argument("--stride", dest="stride", type=int, default=8)
    parser.add_argument("--roi-cutoff", dest="roi_cutoff", type=float, default=.9)
    parser.add_argument("--bkg-cutoff", dest="bkg_cutoff", type=float, default=.5)
    parser.add_argument("--sample-bkg", dest="sample_bkg", action="store_true")
    parser.add_argument("--no-sample-bkg", dest="sample_bkg", action="store_false")
    parser.set_defaults(sample_bkg=True)
    parser.add_argument("--train-out", dest="train_out", type=str, default="./scratch/train")
    parser.add_argument("--val-out", dest="val_out", type=str, default="./scratch/val")
    parser.add_argument("--test-out", dest="test_out", type=str, default="./scratch/test")
    parser.add_argument("--out-img-ext", dest="out_img_ext", type=str, default='png')
    parser.add_argument("--neg-name", dest="neg_name", type=str, default="benign")
    parser.add_argument("--pos-name", dest="pos_name", type=str, default="malignant")
    parser.add_argument("--bkg-name", dest="bkg_name", type=str, default="background")
    parser.add_argument("--augmentation", dest="augmentation", action="store_true")
    parser.add_argument("--no-augmentation", dest="augmentation", action="store_false")
    parser.set_defaults(augmentation=True)
    parser.add_argument("--load-train-ram", dest="load_train_ram", action="store_true")
    parser.add_argument("--no-load-train-ram", dest="load_train_ram", action="store_false")
    parser.set_defaults(load_train_ram=False)
    parser.add_argument("--load-val-ram", dest="load_val_ram", action="store_true")
    parser.add_argument("--no-load-val-ram", dest="load_val_ram", action="store_false")
    parser.set_defaults(load_val_ram=False)
    parser.add_argument("--top-layer-nb", dest="top_layer_nb", type=int, default=None)
    parser.add_argument("--no-top-layer-nb", dest="top_layer_nb", action="store_const", const=None)
    parser.add_argument("--nb-epoch", dest="nb_epoch", type=int, default=10)
    parser.add_argument("--top-layer-epochs", dest="top_layer_epochs", type=int, default=0)
    parser.add_argument("--all-layer-epochs", dest="all_layer_epochs", type=int, default=0)
    parser.add_argument("--optim", dest="optim", type=str, default="sgd")
    parser.add_argument("--init-lr", dest="init_lr", type=float, default=.01)
    parser.add_argument("--top-layer-multiplier", dest="top_layer_multiplier", type=float, default=.01)
    parser.add_argument("--all-layer-multiplier", dest="all_layer_multiplier", type=float, default=.0001)
    parser.add_argument("--es-patience", dest="es_patience", type=int, default=5)
    parser.add_argument("--lr-patience", dest="lr_patience", type=int, default=2)
    parser.add_argument("--weight-decay2", dest="weight_decay2", type=float, default=.01)
    parser.add_argument("--bias-multiplier", dest="bias_multiplier", type=float, default=.1)
    parser.add_argument("--hidden-dropout2", dest="hidden_dropout2", type=float, default=.0)
    parser.add_argument("--exam-tsv", "-et", dest="exam_tsv", type=str, 
                        default="./metadata/exams_metadata.tsv")
    parser.add_argument("--no-exam-tsv", dest="exam_tsv", action="store_const", const=None)
    parser.add_argument("--img-tsv", "-it", dest="img_tsv", type=str, 
                        default="./metadata/images_crosswalk.tsv")
    parser.add_argument("--out", dest="out", type=str, 
                        default="./modelState/subj_lists.pkl")

    args = parser.parse_args()
    run_opts = dict(
        img_extension=args.img_extension, 
        img_height=args.img_height,
        img_scale=args.img_scale,
        neg_vs_pos_ratio=args.neg_vs_pos_ratio,
        test_size=args.test_size,
        val_size=args.val_size,
        net=args.net,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        stride=args.stride,
        roi_cutoff=args.roi_cutoff,
        bkg_cutoff=args.bkg_cutoff,
        sample_bkg=args.sample_bkg,
        train_out=args.train_out,
        val_out=args.val_out,
        test_out=args.test_out,
        out_img_ext=args.out_img_ext,
        neg_name=args.neg_name,
        pos_name=args.pos_name,
        bkg_name=args.bkg_name,
        augmentation=args.augmentation,
        load_train_ram=args.load_train_ram,
        load_val_ram=args.load_val_ram,
        top_layer_nb=args.top_layer_nb,
        nb_epoch=args.nb_epoch,
        top_layer_epochs=args.top_layer_epochs,
        all_layer_epochs=args.all_layer_epochs,
        optim=args.optim,
        init_lr=args.init_lr,
        top_layer_multiplier=args.top_layer_multiplier,
        all_layer_multiplier=args.all_layer_multiplier,
        es_patience=args.es_patience,
        lr_patience=args.lr_patience,
        weight_decay2=args.weight_decay2,
        bias_multiplier=args.bias_multiplier,
        hidden_dropout2=args.hidden_dropout2,
        exam_tsv=args.exam_tsv,
        img_tsv=args.img_tsv,
        out=args.out,
    )
    print "\n"
    print "img_folder=%s" % (args.img_folder)
    print "dl_state=%s" % (args.dl_state)
    print "best_model=%s" % (args.best_model)
    print "\n>>> Model training options: <<<\n", run_opts, "\n"
    run(args.img_folder, args.dl_state, args.best_model, **run_opts)















