from __future__ import print_function
from keras.layers import (
    Input,
    Activation,
    add,
    Dense,
    Reshape,
    Concatenate,
    concatenate,
    multiply,
    Dropout,
    ZeroPadding3D,
    advanced_activations as aa,
    LeakyReLU
)
from keras.layers import Conv2D, SeparableConv2D, GlobalAveragePooling2D, Conv3D, GlobalMaxPooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.layers import Lambda
# from keras_drop_block import DropBlock2D
import keras.backend as K
from keras.utils.vis_utils import plot_model
import numpy as np
import tensorflow as tf
from keras import regularizers
from keras.engine.topology import Layer

regularizers_l2 = 0.00000

def _shortcut(input, residual):
    # return concatenate([input, residual], axis=1)
    return add([input, residual])

# 2D
def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1), bn=False):
    def f(input):
        if bn:                                                                                                                                                                                                                                                                                                                                                                                                              
            input = BatchNormalization(axis=1)(input)
        activation = Activation('relu')(input)

        return Conv2D(filters=nb_filter, kernel_size=(nb_row, nb_col), strides=subsample, 
                         kernel_regularizer=regularizers.l2(regularizers_l2), padding="same")(activation)                                                                                                                                                                                                                                        
    return f

def ResUnits2D(residual_unit, nb_filter, map_height=16, map_width=8, repetations=1, bn=False):
    def f(input):
        for i in range(repetations): 
            init_subsample = (1, 1)
            input = _residual_unit(nb_filter=nb_filter,
                                  init_subsample=init_subsample,
                                  bn=bn)(input)
            # y = cbam_block(y)                      
            # input = add([input, y])

        return input
    return f

def _residual_unit(nb_filter, init_subsample=(1, 1), bn=False):
    def f(input):
        residual = _bn_relu_conv(nb_filter, 3, 3, bn=bn)(input)
        residual = _bn_relu_conv(nb_filter, 3, 3, bn=bn)(residual)
        return _shortcut(input, residual)
    return f

def STAR(c_conf=(3, 2, 32, 32), p_conf=(1, 2, 32, 32), t_conf=(1, 2, 32, 32), external_dim=8, nb_residual_unit=3, bn=False, bn2=False):
    '''
    C - Temporal Closeness
    P - Period
    T - Trend
    conf = (len_seq, nb_flow, map_height, map_width)
    external_dim
    '''
    # map_height, map_width = 32, 32
    map_height, map_width = c_conf[2], c_conf[3]
    nb_flow = 2
    nb_filter = 64

    main_inputs = []

    input = Input(shape=((nb_flow * (c_conf[0]+p_conf[0]*2+t_conf[0]*2), map_height, map_width)))

    main_inputs.append(input)
    main_output = main_inputs[0]

    if external_dim != None and external_dim > 0:
        # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(units=10, activation='relu')(external_input)
        h1 = Dense(units=2*map_height * map_width, activation='relu')(embedding)
        external_output = Reshape((2, map_height, map_width))(h1)
        main_output = Concatenate(axis=1)([main_output, external_output])
    else:
        print('external_dim:', external_dim)

    conv1 = Conv2D(nb_filter, (3, 3), kernel_regularizer=regularizers.l2(regularizers_l2),padding="same")(main_output)

    # [nb_residual_unit] Residual Units
    residual_output = ResUnits2D(_residual_unit, nb_filter=nb_filter,
                      repetations=nb_residual_unit, bn=bn)(conv1)
    if (bn2):
        residual_output = BatchNormalization(axis=1)(residual_output)
    activation = Activation('relu')(residual_output)

    # conv2 = Conv2D(nb_filter, (3, 3), padding='same')(activation)
    conv2 = Conv2D(nb_flow, (3, 3), padding='same')(activation)
    main_output = Activation('tanh')(conv2)

    # model = Model(input=main_inputs, output=main_output)
    model = Model(main_inputs, main_output)

    return model

if __name__ == '__main__':
    model = STAR(external_dim=8, nb_residual_unit=2)
    plot_model(model, to_file='/home/suhan/wanghn/ST-ResNet.png', show_shapes=True)
    model.summary()
