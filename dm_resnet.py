# This code is originally from: https://github.com/raghakot/keras-resnet
# Modified by Li Shen for DM challenge.
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dropout,
    Dense,
    Flatten
)
from keras.layers.merge import concatenate, add
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import (
    MaxPooling2D,
    # AveragePooling2D,
    GlobalAveragePooling2D
)
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2, l1_l2
from keras import backend as K
# import warnings
# warnings.filterwarnings('error')


if K.image_data_format() == 'channels_last':
    ROW_AXIS = 1
    COL_AXIS = 2
    CHANNEL_AXIS = 3
else:
    CHANNEL_AXIS = 1
    ROW_AXIS = 2
    COL_AXIS = 3


# Helper to build a conv -> BN -> relu block
def _conv_bn_relu(nb_filter, nb_row, nb_col, strides=(1, 1), 
                  weight_decay=.0001, dropout=.0):
    def f(input):
        conv = Conv2D(filters=nb_filter, kernel_size=(nb_row, nb_col), 
                      strides=strides, kernel_initializer="he_normal", 
                      padding="same", kernel_regularizer=l2(weight_decay))(input)
        norm = BatchNormalization(axis=CHANNEL_AXIS)(conv)
        relu = Activation("relu")(norm)
        return Dropout(dropout)(relu)

    return f


# Helper to build a BN -> relu -> conv block
# This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
def _bn_relu_conv(nb_filter, nb_row, nb_col, strides=(1, 1), 
                  weight_decay=.0001, dropout=.0):
    def f(input):
        norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
        activation = Activation("relu")(norm)
        activation = Dropout(dropout)(activation)
        return Conv2D(filters=nb_filter, kernel_size=(nb_row, nb_col), 
                      strides=strides, kernel_initializer="he_normal", 
                      padding="same", 
                      kernel_regularizer=l2(weight_decay))(activation)

    return f


# Adds a shortcut between input and residual block and merges them with "sum"
def _shortcut(input, residual, weight_decay=.0001, dropout=.0, identity=True, 
              strides=(1, 1)):
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    # !!! The dropout argument is just a place holder. 
    # !!! It shall not be applied to identity mapping.
    # stride_width = input._keras_shape[ROW_AXIS] // residual._keras_shape[ROW_AXIS]
    # stride_height = input._keras_shape[COL_AXIS] // residual._keras_shape[COL_AXIS]
    # equal_channels = residual._keras_shape[CHANNEL_AXIS] == input._keras_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    # if stride_width > 1 or stride_height > 1 or not equal_channels:
    if not identity:
        shortcut = Conv2D(filters=residual._keras_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1), strides=strides,
                          kernel_initializer="he_normal", padding="valid", 
                          kernel_regularizer=l2(weight_decay))(input)

    return add([shortcut, residual])


# Builds a residual block with repeating bottleneck blocks.
def _residual_block(block_function, nb_filters, repetitions, is_first_layer=False, **kw_args):
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            identity = True
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            if i == 0:
                identity = False
            input = block_function(nb_filters=nb_filters, 
                                   init_strides=init_strides, 
                                   identity=identity, **kw_args)(input)
        return input

    return f


# Basic 3 X 3 convolution blocks.
# Use for resnet with layers <= 34
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
def basic_block(nb_filters, init_strides=(1, 1), identity=True, **kw_args):
    def f(input):
        conv1 = _bn_relu_conv(nb_filters, 3, 3, strides=init_strides, **kw_args)(input)
        residual = _bn_relu_conv(nb_filters, 3, 3, **kw_args)(conv1)
        return _shortcut(input, residual, identity=identity, 
                         strides=init_strides, **kw_args)

    return f


def basic_block_org(nb_filters, init_strides=(1, 1), identity=True, **kw_args):
    def f(input):
        conv1 = _conv_bn_relu(nb_filters, 3, 3, strides=init_strides, **kw_args)(input)
        residual = _conv_bn_relu(nb_filters, 3, 3, **kw_args)(conv1)
        return _shortcut(input, residual, identity=identity, 
                         strides=init_strides, **kw_args)

    return f


# Bottleneck architecture for > 34 layer resnet.
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
# Returns a final conv layer of nb_filters * 4
def bottleneck(nb_filters, init_strides=(1, 1), identity=True, **kw_args):
    def f(input):
        conv_1_1 = _bn_relu_conv(nb_filters, 1, 1, strides=init_strides, **kw_args)(input)
        conv_3_3 = _bn_relu_conv(nb_filters, 3, 3, **kw_args)(conv_1_1)
        residual = _bn_relu_conv(nb_filters * 4, 1, 1, **kw_args)(conv_3_3)
        return _shortcut(input, residual, identity=identity, 
                         strides=init_strides, **kw_args)

    return f


def bottleneck_org(nb_filters, init_strides=(1, 1), identity=True, **kw_args):
    def f(input):
        conv_1_1 = _conv_bn_relu(nb_filters, 1, 1, strides=init_strides, **kw_args)(input)
        conv_3_3 = _conv_bn_relu(nb_filters, 3, 3, **kw_args)(conv_1_1)
        residual = _conv_bn_relu(nb_filters * 4, 1, 1, **kw_args)(conv_3_3)
        return _shortcut(input, residual, identity=identity, 
                         strides=init_strides, **kw_args)

    return f


class ResNetBuilder(object):

    @staticmethod
    def _shared_conv_layers(input_shape, block_fn, repetitions, nb_init_filter=64,
                            init_filter_size=7, init_conv_stride=2, pool_size=3,
                            pool_stride=2,
                            weight_decay=.0001, inp_dropout=.0, hidden_dropout=.0):
        '''Create shared conv layers for all inputs
        Args:
            pool_size ([int]): set to 0 or False to turn off the first max pooling.
        '''

        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_data_format() == 'channels_last':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        input_ = Input(shape=input_shape)
        dropped = Dropout(inp_dropout)(input_)
        conv1 = _conv_bn_relu(nb_filter=nb_init_filter, 
                              nb_row=init_filter_size, 
                              nb_col=init_filter_size, 
                              strides=(init_conv_stride, init_conv_stride), 
                              weight_decay=weight_decay, dropout=hidden_dropout)(dropped)
        if pool_size:
            pool1 = MaxPooling2D(pool_size=(pool_size, pool_size), 
                                 strides=(pool_stride, pool_stride), 
                                 padding="same")(conv1)
            block = pool1
        else:
            block = conv1

        nb_filters = nb_init_filter
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, nb_filters=nb_filters, repetitions=r, 
                                    is_first_layer=i == 0, 
                                    weight_decay=weight_decay, 
                                    dropout=hidden_dropout)(block)
            nb_filters *= 2

        # Classifier block
        # pool2 = AveragePooling2D(pool_size=(block._keras_shape[ROW_AXIS],
        #                                     block._keras_shape[COL_AXIS]),
        #                          strides=(1, 1))(block)
        # flatten1 = Flatten()(pool2)
        pool2 = GlobalAveragePooling2D()(block)

        return input_, pool2

    @staticmethod
    def l1l2_penalty_reg(alpha=1.0, l1_ratio=0.5):
        '''Calculate L1 and L2 penalties for a Keras layer
        This follows the same formulation as in the R package glmnet and Sklearn
        Args:
            alpha ([float]): amount of regularization.
            l1_ratio ([float]): portion of L1 penalty. Setting to 1.0 equals 
                    Lasso.
        '''
        if l1_ratio == .0:
            return l2(alpha)
        elif l1_ratio == 1.:
            return l1(alpha)
        else:
            return l1_l2(l1_ratio*alpha, 1./2*(1 - l1_ratio)*alpha)

    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions, nb_init_filter=64,
              init_filter_size=7, init_conv_stride=2, pool_size=3, pool_stride=2,
              weight_decay=.0001, alpha=1., l1_ratio=.5, 
              inp_dropout=.0, hidden_dropout=.0):
        """
        Builds a custom ResNet like architecture.
        :param input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)

        :param num_outputs: The number of outputs at final softmax layer

        :param block_fn: The block function to use. This is either :func:`basic_block` or :func:`bottleneck`.
        The original paper used basic_block for layers < 50

        :param repetitions: Number of repetitions of various block units.
        At each block unit, the number of filters are doubled and the input size is halved

        :return: The keras model.
        """

        inputs, flatten_out = ResNetBuilder._shared_conv_layers(
            input_shape, block_fn, repetitions, 
            nb_init_filter=nb_init_filter, init_filter_size=init_filter_size, 
            init_conv_stride=init_conv_stride, 
            pool_size=pool_size, pool_stride=pool_stride, 
            weight_decay=weight_decay, 
            inp_dropout=inp_dropout, hidden_dropout=hidden_dropout)
        enet_penalty = ResNetBuilder.l1l2_penalty_reg(alpha, l1_ratio)
        activation = "softmax" if num_outputs > 1 else "sigmoid"
        dense = Dense(units=num_outputs, kernel_initializer="he_normal", 
                      activation=activation, kernel_regularizer=enet_penalty)(flatten_out)
        model = Model(inputs=inputs, outputs=dense)
        return model


    @classmethod
    def build_resnet_18(cls, input_shape, num_outputs, 
                        nb_init_filter=64, init_filter_size=7, init_conv_stride=2, 
                        pool_size=3, pool_stride=2, 
                        weight_decay=.0001, alpha=1., l1_ratio=.5,
                        inp_dropout=.0, hidden_dropout=.0):
        return cls.build(
            input_shape, num_outputs, basic_block, [2, 2, 2, 2], 
            nb_init_filter=nb_init_filter, init_filter_size=init_filter_size,
            init_conv_stride=init_conv_stride,
            pool_size=pool_size, pool_stride=pool_stride,
            weight_decay=weight_decay, inp_dropout=inp_dropout, 
            hidden_dropout=hidden_dropout)

    @classmethod
    def build_resnet_34(cls, input_shape, num_outputs, 
                        nb_init_filter=64, init_filter_size=7, init_conv_stride=2, 
                        pool_size=3, pool_stride=2, 
                        weight_decay=.0001, alpha=1., l1_ratio=.5,
                        inp_dropout=.0, hidden_dropout=.0):
        return cls.build(
            input_shape, num_outputs, basic_block, [3, 4, 6, 3], 
            nb_init_filter=nb_init_filter, init_filter_size=init_filter_size,
            init_conv_stride=init_conv_stride,
            pool_size=pool_size, pool_stride=pool_stride,
            weight_decay=weight_decay, inp_dropout=inp_dropout, 
            hidden_dropout=hidden_dropout)

    @classmethod
    def build_resnet_38(cls, input_shape, num_outputs, 
                        nb_init_filter=64, init_filter_size=7, init_conv_stride=2, 
                        pool_size=3, pool_stride=2, 
                        weight_decay=.0001, alpha=1., l1_ratio=.5,
                        inp_dropout=.0, hidden_dropout=.0):
        return cls.build(
            input_shape, num_outputs, bottleneck, [3, 6, 3], 
            nb_init_filter=nb_init_filter, init_filter_size=init_filter_size,
            init_conv_stride=init_conv_stride,
            pool_size=pool_size, pool_stride=pool_stride,
            weight_decay=weight_decay, inp_dropout=inp_dropout, 
            hidden_dropout=hidden_dropout)

    @classmethod
    def build_resnet_50(cls, input_shape, num_outputs, 
                        nb_init_filter=64, init_filter_size=7, init_conv_stride=2, 
                        pool_size=3, pool_stride=2, 
                        weight_decay=.0001, alpha=1., l1_ratio=.5,
                        inp_dropout=.0, hidden_dropout=.0):
        return cls.build(
            input_shape, num_outputs, bottleneck, [3, 4, 6, 3], 
            nb_init_filter=nb_init_filter, init_filter_size=init_filter_size,
            init_conv_stride=init_conv_stride,
            pool_size=pool_size, pool_stride=pool_stride,
            weight_decay=weight_decay, inp_dropout=inp_dropout, 
            hidden_dropout=hidden_dropout)

    @classmethod
    def build_resnet_50_org(cls, input_shape, num_outputs, 
                        nb_init_filter=64, init_filter_size=7, init_conv_stride=2, 
                        pool_size=3, pool_stride=2, 
                        weight_decay=.0001, alpha=1., l1_ratio=.5,
                        inp_dropout=.0, hidden_dropout=.0):
        return cls.build(
            input_shape, num_outputs, bottleneck_org, [3, 4, 6, 3], 
            nb_init_filter=nb_init_filter, init_filter_size=init_filter_size,
            init_conv_stride=init_conv_stride,
            pool_size=pool_size, pool_stride=pool_stride,
            weight_decay=weight_decay, inp_dropout=inp_dropout, 
            hidden_dropout=hidden_dropout)

    @classmethod
    def build_resnet_101(cls, input_shape, num_outputs, 
                         nb_init_filter=64, init_filter_size=7, init_conv_stride=2, 
                         pool_size=3, pool_stride=2, 
                         weight_decay=.0001, alpha=1., l1_ratio=.5,
                         inp_dropout=.0, hidden_dropout=.0):
        return cls.build(
            input_shape, num_outputs, bottleneck, [3, 4, 23, 3], 
            nb_init_filter=nb_init_filter, init_filter_size=init_filter_size,
            init_conv_stride=init_conv_stride,
            pool_size=pool_size, pool_stride=pool_stride,
            weight_decay=weight_decay, inp_dropout=inp_dropout, 
            hidden_dropout=hidden_dropout)

    @classmethod
    def build_resnet_152(cls, input_shape, num_outputs, 
                         nb_init_filter=64, init_filter_size=7, init_conv_stride=2, 
                         pool_size=3, pool_stride=2, 
                         weight_decay=.0001, alpha=1., l1_ratio=.5,
                         inp_dropout=.0, hidden_dropout=.0):
        return cls.build(
            input_shape, num_outputs, bottleneck, [3, 8, 36, 3], 
            nb_init_filter=nb_init_filter, init_filter_size=init_filter_size,
            init_conv_stride=init_conv_stride,
            pool_size=pool_size, pool_stride=pool_stride,
            weight_decay=weight_decay, inp_dropout=inp_dropout, 
            hidden_dropout=hidden_dropout)

    @classmethod
    def build_dm_resnet_14(cls, input_shape, num_outputs, 
                           nb_init_filter=64, init_filter_size=7, init_conv_stride=2, 
                           pool_size=3, pool_stride=2, 
                           weight_decay=.0001, alpha=1., l1_ratio=.5,
                           inp_dropout=.0, hidden_dropout=.0):
        return cls.build(
            input_shape, num_outputs, bottleneck, [1, 1, 1, 1], 
            nb_init_filter=nb_init_filter, init_filter_size=init_filter_size,
            init_conv_stride=init_conv_stride,
            pool_size=pool_size, pool_stride=pool_stride,
            weight_decay=weight_decay, inp_dropout=inp_dropout, 
            hidden_dropout=hidden_dropout)

    @classmethod
    def build_dm_resnet_47rb5(cls, input_shape, num_outputs, 
                           nb_init_filter=64, init_filter_size=7, init_conv_stride=2, 
                           pool_size=3, pool_stride=2, 
                           weight_decay=.0001, alpha=1., l1_ratio=.5,
                           inp_dropout=.0, hidden_dropout=.0):
        return cls.build(
            input_shape, num_outputs, bottleneck, [3, 3, 3, 3, 3], 
            nb_init_filter=nb_init_filter, init_filter_size=init_filter_size,
            init_conv_stride=init_conv_stride,
            pool_size=pool_size, pool_stride=pool_stride,
            weight_decay=weight_decay, inp_dropout=inp_dropout, 
            hidden_dropout=hidden_dropout)

    @classmethod
    def build_dm_resnet_56rb6(cls, input_shape, num_outputs, 
                           nb_init_filter=64, init_filter_size=7, init_conv_stride=2, 
                           pool_size=3, pool_stride=2, 
                           weight_decay=.0001, alpha=1., l1_ratio=.5,
                           inp_dropout=.0, hidden_dropout=.0):
        return cls.build(
            input_shape, num_outputs, bottleneck, [3, 3, 3, 3, 3, 3], 
            nb_init_filter=nb_init_filter, init_filter_size=init_filter_size,
            init_conv_stride=init_conv_stride,
            pool_size=pool_size, pool_stride=pool_stride,
            weight_decay=weight_decay, inp_dropout=inp_dropout, 
            hidden_dropout=hidden_dropout)

    @classmethod
    def build_dm_resnet_65rb7(cls, input_shape, num_outputs, 
                           nb_init_filter=64, init_filter_size=7, init_conv_stride=2, 
                           pool_size=3, pool_stride=2, 
                           weight_decay=.0001, alpha=1., l1_ratio=.5,
                           inp_dropout=.0, hidden_dropout=.0):
        return cls.build(
            input_shape, num_outputs, bottleneck, [3, 3, 3, 3, 3, 3, 3], 
            nb_init_filter=nb_init_filter, init_filter_size=init_filter_size,
            init_conv_stride=init_conv_stride,
            pool_size=pool_size, pool_stride=pool_stride,
            weight_decay=weight_decay, inp_dropout=inp_dropout, 
            hidden_dropout=hidden_dropout)


class MultiViewResNetBuilder(ResNetBuilder):
    '''Residual net with two inputs
    '''
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions, nb_init_filter=64,
              init_filter_size=7, init_conv_stride=2, pool_size=3, pool_stride=2,
              weight_decay=.0001, alpha=1., l1_ratio=.5, 
              inp_dropout=.0, hidden_dropout=.0):
        """
        Builds a custom ResNet like architecture.
        :param input_shape: Shall be the input shapes for both CC and MLO views.

        :param num_outputs: The number of outputs at final softmax layer

        :param block_fn: The block function to use. This is either :func:`basic_block` or :func:`bottleneck`.
        The original paper used basic_block for layers < 50

        :param repetitions: Number of repetitions of various block units.
        At each block unit, the number of filters are doubled and the input size is halved

        :return: The keras model.
        """

        # First, define a shared CNN model for both CC and MLO views.
        input_cc, flatten_cc = ResNetBuilder._shared_conv_layers(
            input_shape, block_fn, repetitions, 
            nb_init_filter=nb_init_filter, init_filter_size=init_filter_size, 
            init_conv_stride=init_conv_stride, 
            pool_size=pool_size, pool_stride=pool_stride, 
            weight_decay=weight_decay, 
            inp_dropout=inp_dropout, hidden_dropout=hidden_dropout)
        input_mlo, flatten_mlo = ResNetBuilder._shared_conv_layers(
            input_shape, block_fn, repetitions, 
            nb_init_filter=nb_init_filter, init_filter_size=init_filter_size, 
            init_conv_stride=init_conv_stride, 
            pool_size=pool_size, pool_stride=pool_stride, 
            weight_decay=weight_decay, 
            inp_dropout=inp_dropout, hidden_dropout=hidden_dropout)
        # Then merge the conv representations of the two views.
        merged_repr = concatenate([flatten_cc, flatten_mlo])
        enet_penalty = ResNetBuilder.l1l2_penalty_reg(alpha, l1_ratio)
        activation = "softmax" if num_outputs > 1 else "sigmoid"
        dense = Dense(units=num_outputs, kernel_initializer="he_normal", 
                      activation=activation, kernel_regularizer=enet_penalty)(merged_repr)
        discr_model = Model(inputs=[input_cc, input_mlo], outputs=dense)
        return discr_model


def main():
    model = MultiViewResNetBuilder.build_resnet_50(
        (1, 288, 224), 1, inp_dropout=.2, hidden_dropout=.5)
    model.compile(loss="binary_crossentropy", optimizer="sgd")
    # model.summary()


if __name__ == '__main__':
    main()

