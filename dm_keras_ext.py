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

    def __init__(self, filepath, test_data, nb_test_samples=None, batch_size=None):
        super(DMAucModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.test_data = test_data
        if isinstance(test_data, tuple):
            if batch_size is None:
                raise Exception('batch_size must be specified when ' + \
                                'validation data is loaded into RAM')
        elif nb_test_samples is None:
            raise Exception('nb_test_samples must be specified when ' + \
                            'test_data is a generator')
        self.nb_test_samples = nb_test_samples
        self.batch_size = batch_size
        self.best_epoch = 0
        self.best_auc = 0.

    def on_epoch_end(self, epoch, logs={}):
        if isinstance(self.test_data, tuple):
            y_true = self.test_data[1]
            y_pred = self.model.predict(self.test_data[0], self.batch_size)
        else:
            self.test_data.reset()
            samples_seen = 0
            y_list = []
            pred_list = []
            while samples_seen < self.nb_test_samples:
                X, y = next(self.test_data)
                samples_seen += len(y)
                y_list.append(y)
                pred_list.append(self.model.predict_on_batch(X))
            y_true = np.concatenate(y_list)
            y_pred = np.concatenate(pred_list)
        if len(np.unique(y_true)) == 1:
            auc = 0.
        else:
            auc = roc_auc_score(y_true, y_pred)
        print " - Epoch:%d, AUROC: %.4f" % (epoch + 1, auc)
        if auc > self.best_auc:
            self.best_epoch = epoch + 1
            self.best_auc = auc
            if self.filepath != "NOSAVE":
                self.model.save(self.filepath)

    def on_train_end(self, logs={}):
        print ">>> Found best AUROC: %.4f at epoch: %d, saved to: %s <<<" % \
            (self.best_auc, self.best_epoch, self.filepath)

