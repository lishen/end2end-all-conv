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

