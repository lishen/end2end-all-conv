import numpy as np
from keras.callbacks import Callback
import keras.backend as K
from sklearn.metrics import roc_auc_score

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
        self.best_epoch = 0
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
        if len(np.unique(y_true)) == 1:
            auc = 0.
        else:
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

