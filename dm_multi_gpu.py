# Originally from: https://github.com/kuza55/keras-extras/blob/master/utils/multi_gpu.py
from keras.layers.merge import concatenate
from keras.layers.core import Lambda
from keras.models import Model
import tensorflow as tf

def make_parallel(model, gpu_count):
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat(0, [ shape[:1]/parts, shape[1:] ])
        stride = tf.concat(0, [ shape[:1]/parts, shape[1:]*0 ])
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in xrange(len(model.outputs)):
        outputs_all.append([])

    #Place a copy of the model on each GPU, each getting a slice of the batch
    # NUM_GPU_DEVICES: number of GPU devices
    # GPUS: list of GPU devices (e.g. "/dev/nvidia2,/dev/nvidia3")
    for i in xrange(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                #Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
                    inputs.append(slice_n)                

                outputs = model(inputs)
                
                if not isinstance(outputs, list):
                    outputs = [outputs]
                
                #Save all the outputs for merging back together later
                for l in xrange(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            if gpu_count > 1:
                merged.append(concatenate(outputs, axis=0))
            else:
                merged.append(outputs[0])
            
        new_model = Model(inputs=model.inputs, outputs=merged)
        funcType = type(model.save)
        # monkeypatch the save to save just the underlying model
        def new_save(self_, filepath, overwrite=True):
            model.save(filepath, overwrite)
        new_model.save = funcType(new_save, new_model)
        return new_model, model
