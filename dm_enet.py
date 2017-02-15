import os
import pickle
from keras.models import load_model, Model
from dm_multi_gpu import make_parallel


class MultiViewDLElasticNet(object):
    '''Make predictions using elastic net based on multi-view DL
    '''
    def __init__(self, dl_state, enet_state):
        dl_model = load_model(dl_state)
        self.repr_model = Model(
                input=dl_model.input, 
                output=dl_model.get_layer(index=-2).output)
        # gpu_count = int(os.getenv('NUM_GPU_DEVICES', 1))
        # self.repr_model = make_parallel(repr_model, gpu_count) \
        #         if gpu_count > 1 else repr_model
        with open(enet_state) as f:
            self.enet_model = pickle.load(f)

    def predict_on_batch(self, x):
        dl_repr = self.repr_model.predict_on_batch(x)
        pred = self.enet_model.predict_proba(dl_repr)[:, 1]
        return pred


class DLRepr(object):
    '''Extract the last hidden layer representation for a DL model
    '''
    def __init__(self, dl_state, custom_objects=None):
        dl_model = load_model(dl_state, custom_objects=custom_objects)
        self.repr_model = Model(input=dl_model.input, 
                                output=dl_model.get_layer(index=-2).output)

    def predict_on_batch(self, x):
        return self.repr_model.predict_on_batch(x)

    def predict(self, x, batch_size=32, verbose=0):
        return self.repr_model.predict(x, batch_size=batch_size, 
                                       verbose=verbose)

    def predict_generator(self, generator, val_samples, max_q_size=10, 
                          nb_worker=1, pickle_safe=False):
        return self.repr_model.predict_generator(
            generator, val_samples, max_q_size=max_q_size,
            nb_worker=nb_worker, pickle_safe=pickle_safe)        








