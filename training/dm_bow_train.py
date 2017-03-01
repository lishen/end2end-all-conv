import argparse
import os, sys
import pickle
import numpy as np
from numpy.random import RandomState
from scipy.sparse import lil_matrix
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from keras.models import load_model
from dm_enet import DLRepr
from meta import DMMetaManager
from dm_image import DMImageDataGenerator
from dm_keras_ext import DMMetrics as dmm

import warnings
import exceptions
warnings.filterwarnings('ignore', category=exceptions.UserWarning)


def sample_roi_repr(roi_generator, sample_per_batch, nb_samples, repr_model, 
                    batch_size=32, random_seed=12345, q_size=20):
    '''Sample candidate ROIs and then extract their DL representations
    '''
    samples_seen = 0
    repr_list = []
    roi_q = []  # a queue for candid ROIs before they are scored.
    while samples_seen < nb_samples:
        rng = RandomState(samples_seen + random_seed)
        X,w = roi_generator.next()
        w /= w.sum()
        ri = rng.choice(len(X), sample_per_batch, replace=False, p=w)
        roi_q.append(X[ri])
        samples_seen += len(ri)
        if len(roi_q) >= q_size:
            X_q = np.concatenate(roi_q)
            repr_list.append(repr_model.predict(X_q, batch_size=batch_size))
            roi_q = []
    if len(roi_q) > 0:
        X_q = np.concatenate(roi_q)
        repr_list.append(repr_model.predict(X_q, batch_size=batch_size))
        roi_q = []
    return np.concatenate(repr_list)


def get_exam_bow_dat(exam_list, nb_words, roi_per_img,
                     img_list=None, prob_out=None, clust_list=None,
                     imgen=None, clf_list=None, transformer=None, **kw_args):
    '''Get the BoW count matrix for an exam list
    '''
    if img_list is not None:
        if prob_out is None or clust_list is None:
            raise Exception("When img_list is not None, [prob_out, clust_list]"
                            " must not be None")
        img_idx_tab = dict(zip(img_list, range(len(img_list))))
    elif imgen is None or clf_list is None:
        raise Exception("When img_list is None, [imgen, clf_list] must not"
                        " be None")
    else:
        pass

    #####################################################
    def get_prob_repr(
        case_all_imgs, target_height, target_scale, 
        img_per_batch, roi_size, 
        low_int_threshold, blob_min_area, blob_min_int, blob_max_int, 
        blob_th_step, 
        seed, dlrepr_model):
        '''Get prob and DL representations for all ROIs for all images of a case
        '''
        roi_generator = imgen.flow_from_candid_roi(
            case_all_imgs,
            target_height=target_height, target_scale=target_scale,
            class_mode=None, validation_mode=True, 
            img_per_batch=img_per_batch, roi_per_img=roi_per_img, 
            roi_size=roi_size,
            low_int_threshold=low_int_threshold, blob_min_area=blob_min_area, 
            blob_min_int=blob_min_int, blob_max_int=blob_max_int, 
            blob_th_step=blob_th_step,
            roi_clf=None, seed=seed)

        pred = dlrepr_model.predict_generator(
            roi_generator, val_samples=roi_per_img*len(case_all_imgs))
        # Split representation and prob.
        dl_repr = pred[0]
        dl_repr = dl_repr.reshape((-1,dl_repr.shape[-1]))  # flatten feature maps.
        prob_ = pred[1]
        if prob_.shape[1] == 3:
            prob_ = prob_[:, 1]  # cancer class prob.
        prob_ = prob_.reshape((len(case_all_imgs),-1))  # img x roi prob
        return prob_, dl_repr

    def get_prob_out(case_all_imgs):
        iis = np.array([ img_idx_tab[img] for img in case_all_imgs])
        return prob_out[iis]

    def get_clust_labs(case_all_imgs, clust):
        iis = np.array([ img_idx_tab[img] for img in case_all_imgs])
        return clust[iis].ravel()  # flattened clust labs for all imgs.

    def get_breast_prob_clust(case_all_imgs):
        '''Get prob and clust labs for all codebooks for one breast
        '''
        if img_list is not None:
            prob_ = get_prob_out(case_all_imgs)
            clust_ = [ get_clust_labs(case_all_imgs, clust) 
                       for clust in clust_list]
        else:
            prob_, roi_repr = get_prob_repr(case_all_imgs, **kw_args)
            if transformer is not None:
                roi_repr = transformer.transform(roi_repr)
            clust_ = [ clf.predict(roi_repr).ravel() 
                       for clf in clf_list]
        return prob_, clust_
    #####################################################


    bow_list = [ lil_matrix((len(exam_list)*2, n), dtype='uint16') 
                 for n in nb_words]
    meta_prob_list = []
    ri = 0
    for subj, exidx, exam in exam_list:
        try:
            cancerL = int(exam['L']['cancer'])
        except ValueError:
            cancerL = 0
        try:
            cancerR = int(exam['R']['cancer'])
        except ValueError:
            cancerR = 0

        try:
            probL, clustL = get_breast_prob_clust(exam['L']['img'])
            for i,bow in enumerate(bow_list):
                for ci in clustL[i]:
                    bow[ri, ci] += 1
        except KeyError:  # unimaged breast.
            # import pdb; pdb.set_trace()
            probL = np.array([[.0]*roi_per_img])
        ri += 1
        meta_prob_list.append((subj, exidx, 'L', cancerL, probL))

        try:
            probR, clustR = get_breast_prob_clust(exam['R']['img'])
            for i,bow in enumerate(bow_list):
                for ci in clustR[i]:
                    bow[ri, ci] += 1
        except KeyError:  # unimaged breast.
            probR = np.array([[.0]*roi_per_img])
        ri += 1
        meta_prob_list.append((subj, exidx, 'R', cancerR, probR))

    return meta_prob_list, bow_list


def run(img_folder, dl_state, img_extension='dcm', 
        img_height=1024, img_scale=4095, val_size=.2, neg_vs_pos_ratio=10., 
        do_featurewise_norm=True, featurewise_mean=873.6, featurewise_std=739.3,
        img_per_batch=2, roi_per_img=32, roi_size=(256, 256), 
        low_int_threshold=.05, blob_min_area=3, 
        blob_min_int=.5, blob_max_int=.85, blob_th_step=10,
        layer_name=['flatten_1', 'dense_1'], layer_index=None,
        roi_state=None, roi_clf_bs=32, 
        pc_components=.95, pc_whiten=True,
        nb_words=[512], km_max_iter=100, km_bs=1000, km_patience=20, km_init=10,
        exam_tsv='./metadata/exams_metadata.tsv',
        img_tsv='./metadata/images_crosswalk.tsv',
        pca_km_states='./modelState/dlrepr_pca_km_models.pkl',
        bow_train_out='./modelState/bow_dat_train.pkl',
        bow_test_out='./modelState/bow_dat_test.pkl'):
    '''Calculate bag of deep visual words count matrix for all breasts
    '''

    # Read some env variables.
    random_seed = int(os.getenv('RANDOM_SEED', 12345))
    rng = RandomState(random_seed)  # an rng used across board.

    # Load and split image and label lists.
    meta_man = DMMetaManager(exam_tsv=exam_tsv, 
                             img_tsv=img_tsv, 
                             img_folder=img_folder, 
                             img_extension=img_extension)
    subj_list, subj_labs = meta_man.get_subj_labs()
    subj_train, subj_test, labs_train, labs_test = train_test_split(
        subj_list, subj_labs, test_size=val_size, stratify=subj_labs, 
        random_state=random_seed)
    if neg_vs_pos_ratio is not None:
        def subset_subj(subj, labs):
            subj = np.array(subj)
            labs = np.array(labs)
            pos_idx = np.where(labs==1)[0]
            neg_idx = np.where(labs==0)[0]
            nb_neg_desired = int(len(pos_idx)*neg_vs_pos_ratio)
            if nb_neg_desired >= len(neg_idx):
                return subj.tolist()
            else:
                neg_chosen = rng.choice(neg_idx, nb_neg_desired, replace=False)
                subset_idx = np.concatenate([pos_idx, neg_chosen])
                return subj[subset_idx].tolist()

        subj_train = subset_subj(subj_train, labs_train)
        subj_test = subset_subj(subj_test, labs_test)

    img_list, lab_list = meta_man.get_flatten_img_list(subj_train)
    lab_list = np.array(lab_list)
    print "Train set - Nb of positive images: %d, Nb of negative images: %d" \
            % ( (lab_list==1).sum(), (lab_list==0).sum())
    sys.stdout.flush()

    # Create image generator for ROIs for representation extraction.
    print "Create an image generator for ROIs"; sys.stdout.flush()
    if do_featurewise_norm:
        imgen = DMImageDataGenerator(
            featurewise_center=True, 
            featurewise_std_normalization=True)
        imgen.mean = featurewise_mean
        imgen.std = featurewise_std
    else:
        imgen = DMImageDataGenerator(
            samplewise_center=True, 
            samplewise_std_normalization=True)

    # Load ROI classifier.
    if roi_state is not None:
        print "Load ROI classifier"; sys.stdout.flush()
        roi_clf = load_model(
            roi_state, 
            custom_objects={
                'sensitivity': dmm.sensitivity, 
                'specificity': dmm.specificity
            }
        )
        graph = tf.get_default_graph()
    else:
        roi_clf = None
        graph = None

    # Create ROI generators for pos and neg images separately.
    print "Create ROI generators for pos and neg images"
    sys.stdout.flush()
    roi_generator = imgen.flow_from_candid_roi(
        img_list, target_height=img_height, target_scale=img_scale,
        class_mode=None, validation_mode=True, 
        img_per_batch=img_per_batch, roi_per_img=roi_per_img, 
        roi_size=roi_size,
        low_int_threshold=low_int_threshold, blob_min_area=blob_min_area, 
        blob_min_int=blob_min_int, blob_max_int=blob_max_int, 
        blob_th_step=blob_th_step,
        tf_graph=graph, roi_clf=roi_clf, clf_bs=roi_clf_bs, 
        return_sample_weight=False, seed=random_seed)

    # Generate image patches and extract their DL representations.
    print "Load DL representation model"; sys.stdout.flush()
    dlrepr_model = DLRepr(
        dl_state,
        custom_objects={
                'sensitivity': dmm.sensitivity, 
                'specificity': dmm.specificity
        },
        layer_name=layer_name, 
        layer_index=layer_index)
    last_output_size = dlrepr_model.get_output_shape()[-1][-1]
    if last_output_size != 3 and last_output_size != 1:
        raise Exception("The last output must be prob outputs (size=3 or 1)")

    nb_tot_samples = len(img_list)*roi_per_img
    print "Extract ROIs from pos and neg images"; sys.stdout.flush()
    pred = dlrepr_model.predict_generator(roi_generator, 
                                          val_samples=nb_tot_samples)
    for i,d in enumerate(pred):
        print "Shape of representation/output data %d:" % (i), d.shape
    sys.stdout.flush()

    # Flatten feature maps, e.g. an 8x8 feature map will become a 64-d vector.
    pred = [ d.reshape((-1,d.shape[-1])) for d in pred]
    for i,d in enumerate(pred):
        print "Shape of flattened data %d:" % (i), d.shape
    sys.stdout.flush()

    # Split representations and prob outputs.
    dl_repr = pred[0]
    prob_out = pred[1]
    if prob_out.shape[1] == 3:
        prob_out = prob_out[:, 1]  # pos class.
    prob_out = prob_out.reshape((len(img_list),-1))
    print "Reshape prob output to:", prob_out.shape; sys.stdout.flush()

    # Use PCA to reduce dimension of the representation data.
    if pc_components is not None:
        print "Start PCA dimension reduction on DL representation"
        sys.stdout.flush()
        pca = PCA(n_components=pc_components, whiten=pc_whiten)
        pca.fit(dl_repr)
        print "Nb of PCA components:", pca.n_components_
        print "Total explained variance ratio: %.4f" % \
                (pca.explained_variance_ratio_.sum())
        dl_repr_pca = pca.transform(dl_repr)
        print "Shape of transformed representation data:", dl_repr_pca.shape
        sys.stdout.flush()
    else:
        pca = None

    # Use K-means to create a codebook for deep visual words.
    print "Start K-means training on DL representation"
    sys.stdout.flush()
    clf_list = []
    clust_list = []
    # Shuffling indices for mini-batches learning.
    perm_idx = rng.permutation(len(dl_repr))
    for n in nb_words:
        print "Train K-means with %d cluster centers" % (n)
        sys.stdout.flush()
        clf = MiniBatchKMeans(n_clusters=n, init='k-means++', 
                              max_iter=km_max_iter, batch_size=km_bs, 
                              compute_labels=True, random_state=random_seed, 
                              tol=0.0, max_no_improvement=km_patience, 
                              init_size=None, n_init=km_init, 
                              reassignment_ratio=0.01, verbose=0)
        clf.fit(dl_repr[perm_idx])
        clf_list.append(clf)
        clust = np.zeros_like(clf.labels_)
        clust[perm_idx] = clf.labels_
        clust = clust.reshape((len(img_list),-1))
        clust_list.append(clust)

    if pca is not None:
        print "Start K-means training on transformed representation"
        sys.stdout.flush()
        clf_list_pca = []
        clust_list_pca = []
        # Shuffling indices for mini-batches learning.
        perm_idx = rng.permutation(len(dl_repr_pca))
        for n in nb_words:
            print "Train K-means with %d cluster centers" % (n)
            sys.stdout.flush()
            clf = MiniBatchKMeans(n_clusters=n, init='k-means++', 
                                  max_iter=km_max_iter, batch_size=km_bs, 
                                  compute_labels=True, random_state=random_seed, 
                                  tol=0.0, max_no_improvement=km_patience, 
                                  init_size=None, n_init=km_init, 
                                  reassignment_ratio=0.01, verbose=0)
            clf.fit(dl_repr_pca[perm_idx])
            clf_list_pca.append(clf)
            clust = np.zeros_like(clf.labels_)
            clust[perm_idx] = clf.labels_
            clust = clust.reshape((len(img_list),-1))
            clust_list_pca.append(clust)


    # Read exam lists.
    exam_train = meta_man.get_flatten_exam_list(
        subj_train, flatten_img_list=True)
    exam_test = meta_man.get_flatten_exam_list(
        subj_test, flatten_img_list=True)
    exam_labs_train = np.array(meta_man.exam_labs(exam_train))
    exam_labs_test = np.array(meta_man.exam_labs(exam_test))
    nb_pos_exams_train = (exam_labs_train==1).sum()
    nb_neg_exams_train = (exam_labs_train==0).sum()
    nb_pos_exams_test = (exam_labs_test==1).sum()
    nb_neg_exams_test = (exam_labs_test==0).sum()
    print "Train set - Nb of pos exams: %d, Nb of neg exams: %d" % \
            (nb_pos_exams_train, nb_neg_exams_train)
    print "Test set - Nb of pos exams: %d, Nb of neg exams: %d" % \
            (nb_pos_exams_test, nb_neg_exams_test)

    # Do BoW counts for each breast.
    print "BoW counting for train exam list"; sys.stdout.flush()
    bow_dat_train = get_exam_bow_dat(
        exam_train, nb_words, roi_per_img,
        img_list=img_list, prob_out=prob_out, clust_list=clust_list)
    for i,d in enumerate(bow_dat_train[1]):
        print "Shape of train BoW matrix %d:" % (i), d.shape
    sys.stdout.flush()

    print "BoW counting for test exam list"; sys.stdout.flush()
    bow_dat_test = get_exam_bow_dat(
        exam_test, nb_words, roi_per_img,
        imgen=imgen, clf_list=clf_list, transformer=None,
        target_height=img_height, target_scale=img_scale,
        img_per_batch=img_per_batch, roi_size=roi_size,
        low_int_threshold=low_int_threshold, blob_min_area=blob_min_area, 
        blob_min_int=blob_min_int, blob_max_int=blob_max_int, 
        blob_th_step=blob_th_step, seed=random_seed, 
        dlrepr_model=dlrepr_model)
    for i,d in enumerate(bow_dat_test[1]):
        print "Shape of test BoW matrix %d:" % (i), d.shape
    sys.stdout.flush()

    if pca is not None:
        print "== Do same BoW counting on PCA transformed data =="
        print "BoW counting for train exam list"; sys.stdout.flush()
        bow_dat_train_pca = get_exam_bow_dat(
            exam_train, nb_words, roi_per_img,
            img_list=img_list, prob_out=prob_out, clust_list=clust_list_pca)
        for i,d in enumerate(bow_dat_train_pca[1]):
            print "Shape of train BoW matrix %d:" % (i), d.shape
        sys.stdout.flush()

        print "BoW counting for test exam list"; sys.stdout.flush()
        bow_dat_test_pca = get_exam_bow_dat(
            exam_test, nb_words, roi_per_img,
            imgen=imgen, clf_list=clf_list_pca, transformer=pca,
            target_height=img_height, target_scale=img_scale,
            img_per_batch=img_per_batch, roi_size=roi_size,
            low_int_threshold=low_int_threshold, blob_min_area=blob_min_area, 
            blob_min_int=blob_min_int, blob_max_int=blob_max_int, 
            blob_th_step=blob_th_step, seed=random_seed, 
            dlrepr_model=dlrepr_model)
        for i,d in enumerate(bow_dat_test_pca[1]):
            print "Shape of test BoW matrix %d:" % (i), d.shape
        sys.stdout.flush()


    # Save K-means model and BoW count data.
    if pca is None:
        pickle.dump(clf_list, open(pca_km_states, 'w'))
        pickle.dump(bow_dat_train, open(bow_train_out, 'w'))
        pickle.dump(bow_dat_test, open(bow_test_out, 'w'))
    else:
        pickle.dump((pca, clf_list), open(pca_km_states, 'w'))
        pickle.dump((bow_dat_train, bow_dat_train_pca), open(bow_train_out, 'w'))
        pickle.dump((bow_dat_test, bow_dat_test_pca), open(bow_test_out, 'w'))

    print "Done."


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="DM BoW training")
    parser.add_argument("img_folder", type=str)
    parser.add_argument("dl_state", type=str)
    parser.add_argument("--img-extension", "-ext", dest="img_extension", type=str, default="dcm")
    parser.add_argument("--img-height", "-ih", dest="img_height", type=int, default=1024)
    parser.add_argument("--img-scale", "-ic", dest="img_scale", type=int, default=4095)
    parser.add_argument("--val-size", "-vs", dest="val_size", type=float, default=.2)
    parser.add_argument("--neg-vs-pos-ratio", dest="neg_vs_pos_ratio", type=float, default=10.)
    parser.add_argument("--featurewise-norm", dest="do_featurewise_norm", action="store_true")
    parser.add_argument("--no-featurewise-norm", dest="do_featurewise_norm", action="store_false")
    parser.set_defaults(do_featurewise_norm=True)
    parser.add_argument("--featurewise-mean", dest="featurewise_mean", type=float, default=873.6)
    parser.add_argument("--featurewise-std", dest="featurewise_std", type=float, default=739.3)
    parser.add_argument("--img-per-batch", "-ipb", dest="img_per_batch", type=int, default=2)
    parser.add_argument("--roi-per-img", "-rpi", dest="roi_per_img", type=int, default=32)
    parser.add_argument("--roi-size", dest="roi_size", nargs=2, type=int, default=[256, 256])
    parser.add_argument("--low-int-threshold", dest="low_int_threshold", type=float, default=.05)
    parser.add_argument("--blob-min-area", dest="blob_min_area", type=int, default=3)
    parser.add_argument("--blob-min-int", dest="blob_min_int", type=float, default=.5)
    parser.add_argument("--blob-max-int", dest="blob_max_int", type=float, default=.85)
    parser.add_argument("--blob-th-step", dest="blob_th_step", type=int, default=10)
    parser.add_argument("--roi-state", dest="roi_state", type=str, default=None)
    parser.add_argument("--no-roi-state", dest="roi_state", action="store_const", const=None)
    parser.add_argument("--roi-clf-bs", dest="roi_clf_bs", type=int, default=32)
    parser.add_argument("--pc-components", dest="pc_components", type=float, default=.95)
    parser.add_argument("--no-pc-components", dest="pc_components", 
                        action="store_const", const=None)
    parser.add_argument("--pc-whiten", dest="pc_whiten", action="store_true")
    parser.add_argument("--no-pc-whiten", dest="pc_whiten", action="store_false")
    parser.set_defaults(pc_whiten=True)
    parser.add_argument("--layer-name", dest="layer_name", nargs=2, type=str, 
                        default=["flatten_1", "dense_1"])
    parser.add_argument("--layer-index", dest="layer_index", nargs=2, type=int, default=None)
    parser.add_argument("--nb-words", dest="nb_words", nargs="+", type=int, default=[512])
    parser.add_argument("--km-max-iter", dest="km_max_iter", type=int, default=100)
    parser.add_argument("--km-bs", dest="km_bs", type=int, default=1000)
    parser.add_argument("--km-patience", dest="km_patience", type=int, default=20)
    parser.add_argument("--km-init", dest="km_init", type=int, default=10)
    parser.add_argument("--exam-tsv", "-et", dest="exam_tsv", type=str, 
                        default="./metadata/exams_metadata.tsv")
    parser.add_argument("--no-exam-tsv", dest="exam_tsv", action="store_const", const=None)
    parser.add_argument("--img-tsv", "-it", dest="img_tsv", type=str, 
                        default="./metadata/images_crosswalk.tsv")
    parser.add_argument("--pca-km-states", dest="pca_km_states", type=str, 
                        default="./modelState/dlrepr_pca_km_models.pkl")
    parser.add_argument("--bow-train-out", dest="bow_train_out", type=str, 
                        default="./modelState/bow_dat_train.pkl")
    parser.add_argument("--bow-test-out", dest="bow_test_out", type=str, 
                        default="./modelState/bow_dat_test.pkl")

    args = parser.parse_args()
    run_opts = dict(
        img_extension=args.img_extension, 
        img_height=args.img_height,
        img_scale=args.img_scale,
        val_size=args.val_size if args.val_size < 1 else int(args.val_size), 
        neg_vs_pos_ratio=args.neg_vs_pos_ratio,
        do_featurewise_norm=args.do_featurewise_norm,
        featurewise_mean=args.featurewise_mean,
        featurewise_std=args.featurewise_std,
        img_per_batch=args.img_per_batch,
        roi_per_img=args.roi_per_img,
        roi_size=tuple(args.roi_size),
        low_int_threshold=args.low_int_threshold,
        blob_min_area=args.blob_min_area,
        blob_min_int=args.blob_min_int,
        blob_max_int=args.blob_max_int,
        blob_th_step=args.blob_th_step,
        roi_state=args.roi_state,
        roi_clf_bs=args.roi_clf_bs,
        pc_components=(args.pc_components if args.pc_components < 1. 
                       else int(args.pc_components)),
        pc_whiten=args.pc_whiten,
        layer_name=args.layer_name,
        layer_index=args.layer_index,
        nb_words=args.nb_words,
        km_max_iter=args.km_max_iter,
        km_bs=args.km_bs,
        km_patience=args.km_patience,
        km_init=args.km_init,
        exam_tsv=args.exam_tsv,
        img_tsv=args.img_tsv,
        pca_km_states=args.pca_km_states,
        bow_train_out=args.bow_train_out,
        bow_test_out=args.bow_test_out
    )
    print "\n>>> Model training options: <<<\n", run_opts, "\n"
    run(args.img_folder, args.dl_state, **run_opts)















