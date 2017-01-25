import numpy as np

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
