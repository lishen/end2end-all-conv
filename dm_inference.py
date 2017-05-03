import numpy as np
import pandas as pd
from dm_region import prob_heatmap_features

INFER_HEADER_VAL = "subjectId\texamIndex\tlaterality\tconfidence\ttarget\n"
INFER_HEADER = "subjectId\tlaterality\tconfidence\n"

def pred_2view_img_list(cc_img_list, mlo_img_list, model, use_mean=False):
    '''Make predictions for all pairwise combinations of the 2 views
    Returns: a combined score based on the specified function.
    '''
    pred_cc_list = []
    pred_mlo_list = []
    for cc in cc_img_list:
        for mlo in mlo_img_list:
            pred_cc_list.append(cc)
            pred_mlo_list.append(mlo)
    pred_cc = np.stack(pred_cc_list)
    pred_mlo = np.stack(pred_mlo_list)
    preds = model.predict_on_batch([pred_cc, pred_mlo])
    if use_mean:
        pred = preds.mean()
    else:
        pred = preds.max()
    return pred

def make_pred_case(cc_phms, mlo_phms, feature_name, cutoff_list, clf_list,
                   k=2, nb_phm=None, use_mean=False):
    fea_df_list = []
    for cutoff in cutoff_list:
        cc_ben_list = []
        cc_mal_list = []
        mlo_ben_list = []
        mlo_mal_list = []
        cc_fea_list = []
        mlo_fea_list = []
        for cc_phm in cc_phms[:nb_phm]:
            cc_fea_list.append(prob_heatmap_features(cc_phm, cutoff, k))
        for mlo_phm in mlo_phms[:nb_phm]:
            mlo_fea_list.append(prob_heatmap_features(mlo_phm, cutoff, k))
        for cc_fea in cc_fea_list:
            for mlo_fea in mlo_fea_list:
                cc_mal_list.append(cc_fea[0])
                cc_ben_list.append(cc_fea[1])
                mlo_mal_list.append(mlo_fea[0])
                mlo_ben_list.append(mlo_fea[1])
        cc_ben = pd.DataFrame.from_records(cc_ben_list)
        cc_mal = pd.DataFrame.from_records(cc_mal_list)
        mlo_ben = pd.DataFrame.from_records(mlo_ben_list)
        mlo_mal = pd.DataFrame.from_records(mlo_mal_list)
        cc_ben.columns = 'cc_ben_' + cc_ben.columns
        cc_mal.columns = 'cc_mal_' + cc_mal.columns
        mlo_ben.columns = 'mlo_ben_' + mlo_ben.columns
        mlo_mal.columns = 'mlo_mal_' + mlo_mal.columns
        fea_df = pd.concat([cc_ben, cc_mal, mlo_ben, mlo_mal], axis=1)
        try:
            fea_df_list.append(fea_df[feature_name])
        except KeyError:
            fea_df_list.append(fea_df)
    all_fea_df = pd.concat(fea_df_list, axis=1)
    # import pdb; pdb.set_trace()
    if len(clf_list) == 1:
        preds = clf_list[0].predict_proba(all_fea_df.values)[:,1]
    else:
        ens_clf = clf_list[0]
        pred_list = []
        for clf in clf_list[1:]:
            pred_list.append(clf.predict_proba(all_fea_df.values)[:,1])
        pred_mat = np.stack(pred_list, axis=1)
        preds = ens_clf.predict_proba(pred_mat)[:,1]
    if use_mean:
        return preds.mean()
    else:
        return preds.max()

