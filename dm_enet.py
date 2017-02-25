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
    '''Extract the hidden layer representations for a DL model
    '''
    def __init__(self, dl_state, custom_objects=None, 
                 layer_name=None, layer_index=None):
        '''DL representations for images
        Args:
            layer_name ([list]): names for the layers to extract. 
            layer_index ([list]): indices for the layers to extract. index=-2
                    corresponds to the last hidden layer. index=-4 
                    corresponds to the last conv layer (before global 
                    averaging).
        '''
        if layer_name is None and layer_index is None:
            raise Exception("One of [layer_name, layer_index] must be specified")
        dl_model = load_model(dl_state, custom_objects=custom_objects)
        if layer_index is not None:
            output_list = [ dl_model.get_layer(index=idx).output 
                            for idx in layer_index]
        else:
            output_list = [ dl_model.get_layer(name=nm).output 
                            for nm in layer_name]

        self.repr_model = Model(input=dl_model.input, output=output_list)

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

    def get_output_shape(self):
        return self.repr_model.output_shape

    def dl_model_summary(self):
        return self.repr_model.summary()








