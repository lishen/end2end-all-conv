import os, argparse, sys
import numpy as np
from keras.models import load_model, Model
from dm_image import DMImageDataGenerator
from dm_keras_ext import (
    load_dat_ram,
    do_2stage_training,
    DMFlush,
    DMAucModelCheckpoint
)
from dm_resnet import add_top_layers, bottleneck_org
from dm_multi_gpu import make_parallel
import warnings
import exceptions
warnings.filterwarnings('ignore', category=exceptions.UserWarning)


def run(train_dir, val_dir, test_dir, patch_model_state=None, resume_from=None,
        img_size=[1152, 896], img_scale=None, rescale_factor=None,
        featurewise_center=True, featurewise_mean=52.16, 
        equalize_hist=False, augmentation=True,
        class_list=['neg', 'pos'], patch_net='resnet50',
        block_type='resnet', top_depths=[512, 512], top_repetitions=[3, 3], 
        bottleneck_enlarge_factor=4, 
        add_heatmap=False, avg_pool_size=[7, 7], 
        add_conv=True, add_shortcut=False,
        hm_strides=(1,1), hm_pool_size=(5,5),
        fc_init_units=64, fc_layers=2,
        top_layer_nb=None,
        batch_size=64, train_bs_multiplier=.5, 
        nb_epoch=5, all_layer_epochs=20,
        load_val_ram=False, load_train_ram=False,
        weight_decay=.0001, hidden_dropout=.0, 
        weight_decay2=.0001, hidden_dropout2=.0, 
        optim='sgd', init_lr=.01, lr_patience=10, es_patience=25,
        auto_batch_balance=False, pos_cls_weight=1.0, neg_cls_weight=1.0,
        all_layer_multiplier=.1,
        best_model='./modelState/image_clf.h5',
        final_model="NOSAVE"):
    '''Train a deep learning model for image classifications
    '''

    # ======= Environmental variables ======== #
    random_seed = int(os.getenv('RANDOM_SEED', 12345))
    nb_worker = int(os.getenv('NUM_CPU_CORES', 4))
    gpu_count = int(os.getenv('NUM_GPU_DEVICES', 1))

    # ========= Image generator ============== #
    if featurewise_center:
        train_imgen = DMImageDataGenerator(featurewise_center=True)
        val_imgen = DMImageDataGenerator(featurewise_center=True)
        test_imgen = DMImageDataGenerator(featurewise_center=True)
        train_imgen.mean = featurewise_mean
        val_imgen.mean = featurewise_mean
        test_imgen.mean = featurewise_mean
    else:
        train_imgen = DMImageDataGenerator()
        val_imgen = DMImageDataGenerator()
        test_imgen = DMImageDataGenerator()

    # Add augmentation options.
    if augmentation:
        train_imgen.horizontal_flip = True 
        train_imgen.vertical_flip = True
        train_imgen.rotation_range = 25.  # in degree.
        train_imgen.shear_range = .2  # in radians.
        train_imgen.zoom_range = [.8, 1.2]  # in proportion.
        train_imgen.channel_shift_range = 20.  # in pixel intensity values.

    # ================= Model creation ============== #
    if resume_from is not None:
        image_model = load_model(resume_from, compile=False)
    else:
        patch_model = load_model(patch_model_state, compile=False)
        image_model, top_layer_nb = add_top_layers(
            patch_model, img_size, patch_net, block_type, 
            top_depths, top_repetitions, bottleneck_org,
            nb_class=len(class_list), shortcut_with_bn=True, 
            bottleneck_enlarge_factor=bottleneck_enlarge_factor,
            dropout=hidden_dropout, weight_decay=weight_decay,
            add_heatmap=add_heatmap, avg_pool_size=avg_pool_size,
            add_conv=add_conv, add_shortcut=add_shortcut,
            hm_strides=hm_strides, hm_pool_size=hm_pool_size, 
            fc_init_units=fc_init_units, fc_layers=fc_layers)
    if gpu_count > 1:
        image_model, org_model = make_parallel(image_model, gpu_count)
    else:
        org_model = image_model

    # ============ Train & validation set =============== #
    train_bs = int(batch_size*train_bs_multiplier)
    if patch_net != 'yaroslav':
        dup_3_channels = True
    else:
        dup_3_channels = False
    if load_train_ram:
        raw_imgen = DMImageDataGenerator()
        print "Create generator for raw train set"
        raw_generator = raw_imgen.flow_from_directory(
            train_dir, target_size=img_size, target_scale=img_scale, 
            rescale_factor=rescale_factor,
            equalize_hist=equalize_hist, dup_3_channels=dup_3_channels,
            classes=class_list, class_mode='categorical', 
            batch_size=train_bs, shuffle=False)
        print "Loading raw train set into RAM.",
        sys.stdout.flush()
        raw_set = load_dat_ram(raw_generator, raw_generator.nb_sample)
        print "Done."; sys.stdout.flush()
        print "Create generator for train set"
        train_generator = train_imgen.flow(
            raw_set[0], raw_set[1], batch_size=train_bs, 
            auto_batch_balance=auto_batch_balance, 
            shuffle=True, seed=random_seed)
    else:
        print "Create generator for train set"
        train_generator = train_imgen.flow_from_directory(
            train_dir, target_size=img_size, target_scale=img_scale,
            rescale_factor=rescale_factor,
            equalize_hist=equalize_hist, dup_3_channels=dup_3_channels,
            classes=class_list, class_mode='categorical', 
            auto_batch_balance=auto_batch_balance, batch_size=train_bs, 
            shuffle=True, seed=random_seed)

    print "Create generator for val set"
    validation_set = val_imgen.flow_from_directory(
        val_dir, target_size=img_size, target_scale=img_scale,
        rescale_factor=rescale_factor,
        equalize_hist=equalize_hist, dup_3_channels=dup_3_channels,
        classes=class_list, class_mode='categorical', 
        batch_size=batch_size, shuffle=False)
    sys.stdout.flush()
    if load_val_ram:
        print "Loading validation set into RAM.",
        sys.stdout.flush()
        validation_set = load_dat_ram(validation_set, validation_set.nb_sample)
        print "Done."; sys.stdout.flush()

    # ==================== Model training ==================== #
    # Do 2-stage training.
    train_batches = int(train_generator.nb_sample/train_bs) + 1
    if isinstance(validation_set, tuple):
        val_samples = len(validation_set[0])
    else:
        val_samples = validation_set.nb_sample
    validation_steps = int(val_samples/batch_size)
    #### DEBUG ####
    # train_batches = 1
    # val_samples = batch_size*5
    # validation_steps = 5
    #### DEBUG ####
    if load_val_ram:
        auc_checkpointer = DMAucModelCheckpoint(
            best_model, validation_set, batch_size=batch_size)
    else:
        auc_checkpointer = DMAucModelCheckpoint(
            best_model, validation_set, test_samples=val_samples)
    # import pdb; pdb.set_trace()
    image_model, loss_hist, acc_hist = do_2stage_training(
        image_model, org_model, train_generator, validation_set, validation_steps, 
        best_model, train_batches, top_layer_nb, nb_epoch=nb_epoch,
        all_layer_epochs=all_layer_epochs,
        optim=optim, init_lr=init_lr, 
        all_layer_multiplier=all_layer_multiplier,
        es_patience=es_patience, lr_patience=lr_patience, 
        auto_batch_balance=auto_batch_balance, 
        pos_cls_weight=pos_cls_weight, neg_cls_weight=neg_cls_weight,
        nb_worker=nb_worker, auc_checkpointer=auc_checkpointer,
        weight_decay=weight_decay, hidden_dropout=hidden_dropout,
        weight_decay2=weight_decay2, hidden_dropout2=hidden_dropout2,)

    # Training report.
    if len(loss_hist) > 0:
        min_loss_locs, = np.where(loss_hist == min(loss_hist))
        best_val_loss = loss_hist[min_loss_locs[0]]
        best_val_accuracy = acc_hist[min_loss_locs[0]]
        print "\n==== Training summary ===="
        print "Minimum val loss achieved at epoch:", min_loss_locs[0] + 1
        print "Best val loss:", best_val_loss
        print "Best val accuracy:", best_val_accuracy

    if final_model != "NOSAVE":
        image_model.save(final_model)

    # ==== Predict on test set ==== #
    print "\n==== Predicting on test set ===="
    test_generator = test_imgen.flow_from_directory(
        test_dir, target_size=img_size, target_scale=img_scale,
        rescale_factor=rescale_factor,
        equalize_hist=equalize_hist, dup_3_channels=dup_3_channels, 
        classes=class_list, class_mode='categorical', batch_size=batch_size, 
        shuffle=False)
    test_samples = test_generator.nb_sample
    #### DEBUG ####
    # test_samples = 5
    #### DEBUG ####
    print "Test samples =", test_samples
    print "Load saved best model:", best_model + '.',
    sys.stdout.flush()
    org_model.load_weights(best_model)
    print "Done."
    # test_steps = int(test_generator.nb_sample/batch_size)
    # test_res = image_model.evaluate_generator(
    #     test_generator, test_steps, nb_worker=nb_worker, 
    #     pickle_safe=True if nb_worker > 1 else False)
    test_auc = DMAucModelCheckpoint.calc_test_auc(
        test_generator, image_model, test_samples=test_samples)
    print "AUROC on test set:", test_auc


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="DM image clf training")
    parser.add_argument("train_dir", type=str)
    parser.add_argument("val_dir", type=str)
    parser.add_argument("test_dir", type=str)
    parser.add_argument("--patch-model-state", dest="patch_model_state", type=str, default=None)
    parser.add_argument("--no-patch-model-state", dest="patch_model_state", action="store_const", const=None)
    parser.add_argument("--resume-from", dest="resume_from", type=str, default=None)
    parser.add_argument("--no-resume-from", dest="resume_from", action="store_const", const=None)
    parser.add_argument("--img-size", "-is", dest="img_size", nargs=2, type=int, default=[1152, 896])
    parser.add_argument("--img-scale", "-ic", dest="img_scale", type=float, default=None)
    parser.add_argument("--no-img-scale", "-nic", dest="img_scale", action="store_const", const=None)
    parser.add_argument("--rescale-factor", dest="rescale_factor", type=float, default=None)
    parser.add_argument("--no-rescale-factor", dest="rescale_factor", action="store_const", const=None)
    parser.add_argument("--featurewise-center", dest="featurewise_center", action="store_true")
    parser.add_argument("--no-featurewise-center", dest="featurewise_center", action="store_false")
    parser.set_defaults(featurewise_center=True)
    parser.add_argument("--featurewise-mean", dest="featurewise_mean", type=float, default=52.16)
    parser.add_argument("--equalize-hist", dest="equalize_hist", action="store_true")
    parser.add_argument("--no-equalize-hist", dest="equalize_hist", action="store_false")
    parser.set_defaults(equalize_hist=False)
    parser.add_argument("--batch-size", "-bs", dest="batch_size", type=int, default=64)
    parser.add_argument("--train-bs-multiplier", dest="train_bs_multiplier", type=float, default=.5)
    parser.add_argument("--augmentation", dest="augmentation", action="store_true")
    parser.add_argument("--no-augmentation", dest="augmentation", action="store_false")
    parser.set_defaults(augmentation=True)
    parser.add_argument("--class-list", dest="class_list", nargs='+', type=str, 
                        default=['neg', 'pos'])
    parser.add_argument("--patch-net", dest="patch_net", type=str, default="resnet50")
    parser.add_argument("--block-type", dest="block_type", type=str, default="resnet")
    parser.add_argument("--top-depths", dest="top_depths", nargs='+', type=int, default=[512, 512])
    parser.add_argument("--top-repetitions", dest="top_repetitions", nargs='+', type=int, 
                        default=[3, 3])
    parser.add_argument("--bottleneck-enlarge-factor", dest="bottleneck_enlarge_factor", 
                        type=int, default=4)
    parser.add_argument("--add-heatmap", dest="add_heatmap", action="store_true")
    parser.add_argument("--no-add-heatmap", dest="add_heatmap", action="store_false")
    parser.set_defaults(add_heatmap=False)
    parser.add_argument("--avg-pool-size", dest="avg_pool_size", nargs=2, type=int, default=[7, 7])
    parser.add_argument("--add-conv", dest="add_conv", action="store_true")
    parser.add_argument("--no-add-conv", dest="add_conv", action="store_false")
    parser.set_defaults(add_conv=True)
    parser.add_argument("--add-shortcut", dest="add_shortcut", action="store_true")
    parser.add_argument("--no-add-shortcut", dest="add_shortcut", action="store_false")
    parser.set_defaults(add_shortcut=False)
    parser.add_argument("--hm-strides", dest="hm_strides", nargs=2, type=int, default=[1, 1])
    parser.add_argument("--hm-pool-size", dest="hm_pool_size", nargs=2, type=int, default=[5,5])
    parser.add_argument("--fc-init-units", dest="fc_init_units", type=int, default=64)
    parser.add_argument("--fc-layers", dest="fc_layers", type=int, default=2)
    parser.add_argument("--top-layer-nb", dest="top_layer_nb", type=int, default=None)
    parser.add_argument("--no-top-layer-nb", dest="top_layer_nb", action="store_const", const=None)
    parser.add_argument("--nb-epoch", "-ne", dest="nb_epoch", type=int, default=5)
    parser.add_argument("--all-layer-epochs", dest="all_layer_epochs", type=int, default=20)
    parser.add_argument("--load-val-ram", dest="load_val_ram", action="store_true")
    parser.add_argument("--no-load-val-ram", dest="load_val_ram", action="store_false")
    parser.set_defaults(load_val_ram=False)
    parser.add_argument("--load-train-ram", dest="load_train_ram", action="store_true")
    parser.add_argument("--no-load-train-ram", dest="load_train_ram", action="store_false")
    parser.set_defaults(load_train_ram=False)
    parser.add_argument("--weight-decay", "-wd", dest="weight_decay", type=float, default=.0001)
    parser.add_argument("--hidden-dropout", "-hd", dest="hidden_dropout", type=float, default=.0)
    parser.add_argument("--weight-decay2", "-wd2", dest="weight_decay2", type=float, default=.0001)
    parser.add_argument("--hidden-dropout2", "-hd2", dest="hidden_dropout2", type=float, default=.0)
    parser.add_argument("--optimizer", dest="optim", type=str, default="sgd")
    parser.add_argument("--init-learningrate", "-ilr", dest="init_lr", type=float, default=.01)
    parser.add_argument("--lr-patience", "-lrp", dest="lr_patience", type=int, default=10)
    parser.add_argument("--es-patience", "-esp", dest="es_patience", type=int, default=25)
    parser.add_argument("--auto-batch-balance", dest="auto_batch_balance", action="store_true")
    parser.add_argument("--no-auto-batch-balance", dest="auto_batch_balance", action="store_false")
    parser.set_defaults(auto_batch_balance=True)
    parser.add_argument("--pos-cls-weight", dest="pos_cls_weight", type=float, default=1.0)
    parser.add_argument("--neg-cls-weight", dest="neg_cls_weight", type=float, default=1.0)
    parser.add_argument("--all-layer-multiplier", dest="all_layer_multiplier", type=float, default=.1)
    parser.add_argument("--best-model", "-bm", dest="best_model", type=str, 
                        default="./modelState/image_clf.h5")
    parser.add_argument("--final-model", "-fm", dest="final_model", type=str, 
                        default="NOSAVE")

    args = parser.parse_args()
    if args.patch_model_state is None and args.resume_from is None:
        raise Exception('One of [patch_model_state, resume_from] must not be None.')
    run_opts = dict(
        patch_model_state=args.patch_model_state,
        resume_from=args.resume_from,
        img_size=args.img_size, 
        img_scale=args.img_scale, 
        rescale_factor=args.rescale_factor,
        featurewise_center=args.featurewise_center,
        featurewise_mean=args.featurewise_mean,
        equalize_hist=args.equalize_hist,
        batch_size=args.batch_size, 
        train_bs_multiplier=args.train_bs_multiplier,
        augmentation=args.augmentation,
        class_list=args.class_list,
        patch_net=args.patch_net,
        block_type=args.block_type,
        top_depths=args.top_depths,
        top_repetitions=args.top_repetitions,
        bottleneck_enlarge_factor=args.bottleneck_enlarge_factor,
        add_heatmap=args.add_heatmap,
        avg_pool_size=args.avg_pool_size,
        add_conv=args.add_conv,
        add_shortcut=args.add_shortcut,
        hm_strides=args.hm_strides,
        hm_pool_size=args.hm_pool_size,
        fc_init_units=args.fc_init_units,
        fc_layers=args.fc_layers,
        top_layer_nb=args.top_layer_nb,
        nb_epoch=args.nb_epoch, 
        all_layer_epochs=args.all_layer_epochs,
        load_val_ram=args.load_val_ram,
        load_train_ram=args.load_train_ram,
        weight_decay=args.weight_decay,
        hidden_dropout=args.hidden_dropout,
        weight_decay2=args.weight_decay2,
        hidden_dropout2=args.hidden_dropout2,
        optim=args.optim,
        init_lr=args.init_lr,
        lr_patience=args.lr_patience, 
        es_patience=args.es_patience,
        auto_batch_balance=args.auto_batch_balance,
        pos_cls_weight=args.pos_cls_weight,
        neg_cls_weight=args.neg_cls_weight,
        all_layer_multiplier=args.all_layer_multiplier,
        best_model=args.best_model,        
        final_model=args.final_model        
    )
    print "\ntrain_dir=%s" % (args.train_dir)
    print "val_dir=%s" % (args.val_dir)
    print "test_dir=%s" % (args.test_dir)
    print "\n>>> Model training options: <<<\n", run_opts, "\n"
    run(args.train_dir, args.val_dir, args.test_dir, **run_opts)









