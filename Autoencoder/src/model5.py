'''
togliere downsampling
'''

from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Flatten,
    Concatenate,
    Dense,
    Reshape,
    Conv2D,
    BatchNormalization,
    Add,
    Conv2DTranspose,
    LSTM
)
from keras.optimizers import Adam
from keras import regularizers
import numpy as np
import tensorflow as tf
import src.metrics as metrics


def conv_relu_bn(filters, kernel_size=(3,3)):
    def f(input_layer):
        l = Conv2D(filters, kernel_size=kernel_size, padding='same', activation='relu', kernel_regularizer=regularizers.l2(l2=1e-4))(input_layer)
        l = BatchNormalization()(l)
        return l                                                                                                                                                                                                                                      
    return f

def _residual_unit(filters, num_res, kernel_size):
    def f(input_layer):
        residual = conv_relu_bn(filters, kernel_size)(input_layer)
        for _ in range(num_res-1):
            residual = conv_relu_bn(filters, kernel_size)(residual)
        
        return Add()([input_layer, residual])
    return f

def resUnits2D(filters, num_res, kernel_size, repetations=1):
    def f(input_layer):
        for i in range(repetations):
            input_layer = _residual_unit(filters, num_res, kernel_size)(input_layer)
        return input_layer
    return f

def my_downsampling(input_layer):
    l = Conv2D(input_layer.shape[-1], (2,2), (2,2), activation='relu')(input_layer)
    l = BatchNormalization()(l)
    return l

def my_conv_transpose(input_layer, skip_connection_layer):
    # l = Conv2DTranspose(input_layer.shape[-1], (2,2), (2,2))(input_layer)
    l = input_layer
    l = Add()([l, skip_connection_layer])
    l = Activation('relu')(l)
    l = BatchNormalization()(l)
    return l

def my_model(len_c, len_p, len_t, nb_flow=2, map_height=32, map_width=32, external_dim=8, encoder_blocks=3, num_res=1, filters=[32,64,64,16], lstm_units=16, kernel_size=(3,3)):

    main_inputs = []
    #ENCODER
    # input layer 32x32x14
    input = Input(shape=((map_height, map_width, nb_flow * (len_c+len_p*2+len_t*2))))
    main_inputs.append(input)
    x = input

    # merge external features
    if external_dim != None and external_dim > 0:
        # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(units=10, activation='relu')(external_input)
        h1 = Dense(units=nb_flow*map_height * map_width, activation='relu')(embedding)
        external_output = Reshape((map_height, map_width, nb_flow))(h1)
        main_output = Concatenate(axis=3)([input, external_output])
        x = main_output

    # build encoder blocks
    skip_connection_layers = []
    for i in range(0, encoder_blocks):
        # conv + relu + bn + res
        x = conv_relu_bn(filters[i])(x)
        x = resUnits2D(filters[i], num_res, kernel_size)(x)
        # append layer to skip connection list
        skip_connection_layers.append(x)
        # max pool
        # x = my_downsampling(x)

    # last convolution 4x4x16
    x = conv_relu_bn(filters[-1])(x)
    s = x.shape

    x = Reshape((x.shape[1]*x.shape[2], x.shape[3]))(x)
    x = LSTM(lstm_units, return_sequences=True)(x)
    x = LSTM(lstm_units, return_sequences=True)(x)
    x = LSTM(lstm_units, return_sequences=True)(x)
    x = LSTM(filters[-1], return_sequences=True)(x)
    x = Reshape((s[1:]))(x)

    # build decoder blocks
    for i in reversed(range(0, encoder_blocks)):
        # conv + relu + bn
        x = conv_relu_bn(filters[i])(x)
        x = resUnits2D(filters[i], num_res, kernel_size)(x)
        # conv_transpose + skip_conn + relu + bn
        x = my_conv_transpose(x, skip_connection_layers[i])

    # last convolution + tanh + bn 32x32x2
    output = Conv2D(nb_flow, (3,3), padding='same', activation='tanh')(x)

    return Model(main_inputs, output)

def build_model(len_c, len_p, len_t, nb_flow=2, map_height=32, map_width=32,
                external_dim=8, encoder_blocks=3, num_res=1, filters=[32,64,64,16],
                lstm_units=16, lr=0.0001, kernel_size=(3,3), save_model_pic=None):
    model = my_model(len_c, len_p, len_t, nb_flow, map_height, map_width,
                     external_dim, encoder_blocks, num_res, filters, lstm_units, kernel_size)
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    # model.summary()
    if (save_model_pic):
        from keras.utils.vis_utils import plot_model
        plot_model(model, to_file=f'{save_model_pic}.png', show_shapes=True)
    return model
