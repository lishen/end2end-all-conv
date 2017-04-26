import pandas as pd
from meta import DMMetaManager
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

fea_all_k3_nAll_train_mean_val = pd.read_csv('m5_ftu_feaAll_k3_train_mean_pred.tsv', sep="\t")
fea_all_k3_nAll_test_mean_val = pd.read_csv('m5_ftu_feaAll_k3_test_mean_pred.tsv', sep="\t")
subj_train = fea_all_k3_nAll_train_mean_val['subjectId'].unique()
subj_test = fea_all_k3_nAll_test_mean_val['subjectId'].unique()
# => Load exam_df from an external file.
exam_df = pickle.load(open('exam_df.pkl'))
man = DMMetaManager(exam_df=exam_df)
exam_df_train = man.get_flatten_2_exam_dat(subj_train, 'm5_ftu_feaAll_k3_train_mean_pred.tsv')
exam_df_test = man.get_flatten_2_exam_dat(subj_test, 'm5_ftu_feaAll_k3_test_mean_pred.tsv')

# Random forest.
imp = Imputer(missing_values='NaN', strategy='mean')
rf = RandomForestClassifier(n_estimators=50, class_weight='balanced', 
                            random_state=12345, n_jobs=-1)
imp_clf = Pipeline([('imp', imp), ('clf', rf)])

rf_grid_param1 = {
    'clf__min_samples_split': [2, 100, 200, 300],
    'clf__max_depth': range(3, 10, 2),
}

rf_gsearch1 = GridSearchCV(
    estimator=imp_clf,
    param_grid=rf_grid_param1,
    scoring='roc_auc', n_jobs=1, cv=5,
)
rf_gsearch1.fit(exam_df_train[0], exam_df_train[1])

print rf_gsearch1.best_params_
print '='*10
print rf_gsearch1.best_score_
print '='*10
print roc_auc_score(
    exam_df_test[1], rf_gsearch1.predict_proba(exam_df_test[0])[:,1])

pickle.dump(rf_gsearch1.best_estimator_, open('model5_ftu_based_meta_clf.pkl', 'w'))




