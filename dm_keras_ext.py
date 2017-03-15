import sys
import numpy as np
from keras.callbacks import Callback
import keras.backend as K
from sklearn.metrics import roc_auc_score


def load_dat_ram(generator, nb_samples):
    samples_seen = 0
    X_list = []
    y_list = []
    w_list = []
    while samples_seen < nb_samples:
        blob_ = generator.next()
        try:
            X,y,w = blob_
            w_list.append(w)
        except ValueError:
            X,y = blob_
        X_list.append(X)
        y_list.append(y)
        samples_seen += len(y)
    try:
        data_set = (np.concatenate(X_list), 
                    np.concatenate(y_list),
                    np.concatenate(w_list))
    except ValueError:
        data_set = (np.concatenate(X_list), 
                    np.concatenate(y_list))

    if len(data_set[0]) != nb_samples:
        raise Exception('Load data into RAM error')

    return data_set


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

    def __init__(self, filepath, test_data, nb_test_samples=None, 
                 batch_size=None):
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
        self.best_all_auc = None

    def on_epoch_end(self, epoch, logs={}):
        if isinstance(self.test_data, tuple):
            y_true = self.test_data[1]
            y_pred = self.model.predict(self.test_data[0], self.batch_size)
            if len(self.test_data) > 2:
                weights = self.test_data[2]
            else:
                weights = None
        else:
            self.test_data.reset()
            samples_seen = 0
            y_list = []
            pred_list = []
            wei_list = []
            while samples_seen < self.nb_test_samples:
                res = next(self.test_data)
                if len(res) > 2:
                    w = res[2]
                    wei_list.append(w)
                X, y = res[:2]
                samples_seen += len(y)
                y_list.append(y)
                pred_list.append(self.model.predict_on_batch(X))
            y_true = np.concatenate(y_list)
            y_pred = np.concatenate(pred_list)
            if len(wei_list) > 0:
                weights = np.concatenate(wei_list)
            else:
                weights = None
        # Calculate AUC score.
        try:
            auc = roc_auc_score(y_true, y_pred, average=None, 
                                sample_weight=weights)
        except ValueError:
            auc = .0
        # Calculate AUC for pos and neg classes on non-background cases.
        if y_pred.shape[1] == 3:
            non_bkg_idx = np.where(y_true[:,0]==0)[0]
            sample_weight = None if weights is None else weights[non_bkg_idx]
            try:
                non_bkg_auc_pos = roc_auc_score(
                    y_true[non_bkg_idx, 1], y_pred[non_bkg_idx, 1], 
                    sample_weight=sample_weight)
            except ValueError:
                non_bkg_auc_pos = .0
            try:
                non_bkg_auc_neg = roc_auc_score(
                    y_true[non_bkg_idx, 2], y_pred[non_bkg_idx, 2], 
                    sample_weight=sample_weight)
            except ValueError:
                non_bkg_auc_neg = .0
            # import pdb; pdb.set_trace()
        if isinstance(auc, float):
            print " - Epoch:%d, AUROC: %.4f" % (epoch + 1, auc)
        elif len(auc) == 3:
            print " - Epoch:%d, AUROC: bkg - %.4f, pos - %.4f, neg - %.4f" \
                    % (epoch + 1, auc[0], auc[1], auc[2])
            print " - non-bkg pos AUROC: %.4f, neg AUROC: %.4f" \
                    % (non_bkg_auc_pos, non_bkg_auc_neg)
        else:
            raise Exception("Unknown auc format: " + str(auc))
        sys.stdout.flush()
        epoch_auc = non_bkg_auc_pos if y_pred.shape[1] == 3 else auc
        if epoch_auc > self.best_auc:
            self.best_epoch = epoch + 1
            self.best_auc = epoch_auc
            self.best_all_auc = auc
            if self.filepath != "NOSAVE":
                self.model.save(self.filepath)

    def on_train_end(self, logs={}):
        print "\n>>> Found best AUROC: %.4f at epoch: %d, saved to: %s <<<" % \
            (self.best_auc, self.best_epoch, self.filepath)
        print ">>> AUROC for all cls:", str(self.best_all_auc), "<<<"
        sys.stdout.flush()


class DMFlush(Callback):
    '''A callback does nothing but flushes stdout after each epoch
    '''
    def __init__(self):
        super(DMFlush, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        sys.stdout.flush()


