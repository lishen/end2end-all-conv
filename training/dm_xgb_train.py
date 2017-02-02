# !!! This is not a standalone program. Copy&paste to Python console to run. !!! #
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from meta import DMMetaManager


# =========== Load and construct training data =========== #
# meta_man = DMMetaManager(exam_tsv=exam_tsv, img_tsv=img_tsv, 
#                          img_folder=img_folder, img_extension=img_extension)
exam_df = pickle.load(open('exam_df.pkl'))
meta_man = DMMetaManager(exam_df=exam_df)
subj_list, labs_list = man.get_subj_labs()
subj_train, subj_test, labs_train, labs_test = train_test_split(
    subj_list, labs_list, test_size=8000, stratify=labs_list, random_state=12345)
X_train, y_train = man.get_flatten_2_exam_dat(subj_train, 'predictions_max_corrected.tsv')
X_test, y_test = man.get_flatten_2_exam_dat(subj_test, 'predictions_max_corrected.tsv')
# ============= Train xgb ============= #
dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)
param = {'colsample_bytree': 0.5,
         'eta': 0.02,
         'eval_metric': ['logloss', 'auc'],
         'max_depth': 5,
         'min_child_weight': 1,
         'objective': 'binary:logistic',
         'scale_pos_weight': 5,
         'seed': 12345,
         'silent': 1,
         'subsample': 0.8}
num_round = 500
early_stopping_rounds = 20
watchlist = [ (dtrain, 'train'), (dtest, 'eval') ]
bst = xgb.train(param, dtrain, num_round, watchlist, 
                early_stopping_rounds=early_stopping_rounds)
test_pred = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, test_pred)
# ============ Feature importance ============== #
xgb.plot_importance(bst, importance_type='weight')
import matplotlib.pyplot as plt
plt.show()
# ============ Save model ============== #
pickle.dump(bst, open('xgb_2017-01-25-10am/bst_model.pkl', 'w'))
np.savez_compressed(
    'xgb_2017-01-25-10am/xgb_param.npz', 
    early_stopping_rounds=early_stopping_rounds, 
    num_round=num_round, param=param)
