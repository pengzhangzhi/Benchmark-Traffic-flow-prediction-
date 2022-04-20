'''
    ST-ResNet: Deep Spatio-temporal Residual Networks
'''

from __future__ import print_function
import keras
from keras.layers import (
    Input,
    Activation,
    Add,
    Dense,
    Reshape,
    Conv2D,
    BatchNormalization
)
from keras.models import Model
#from keras.utils.visualize_util import plot


def _shortcut(input, residual):
    return Add()([input, residual])


def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1), bn=False):
    def f(input):
        if bn:
            input = BatchNormalization(axis=1)(input)
        activation = Activation('relu')(input)
        return Conv2D(nb_filter, (nb_row, nb_col), subsample, padding="same")(activation)
    return f


def _residual_unit(nb_filter, init_subsample=(1, 1), bn=False):
    def f(input):
        residual = _bn_relu_conv(nb_filter, 3, 3, bn=bn)(input)
        residual = _bn_relu_conv(nb_filter, 3, 3, bn=bn)(residual)
        return _shortcut(input, residual)
    return f


def ResUnits(residual_unit, nb_filter, repetations=1, bn=False):
    def f(input):
        for i in range(repetations):
            init_subsample = (1, 1)
            input = residual_unit(nb_filter=nb_filter,
                                  init_subsample=init_subsample,
                                  bn=bn)(input)
        return input
    return f


def stresnet(c_conf=(3, 2, 32, 32), p_conf=(3, 2, 32, 32), t_conf=(3, 2, 32, 32), external_dim=8, nb_residual_unit=3, bn=False, bn2=False):
    '''
    C - Temporal Closeness
    P - Period
    T - Trend
    conf = (len_seq, nb_flow, map_height, map_width)
    external_dim
    '''
    keras.backend.set_image_data_format('channels_first')
    # main input
    main_inputs = []
    outputs = []
    for conf in [c_conf, p_conf, t_conf]:
        if conf is not None:
            len_seq, nb_flow, map_height, map_width = conf
            input = Input(shape=(nb_flow * len_seq, map_height, map_width))
            main_inputs.append(input)
            # Conv1
            conv1 = Conv2D(
                filters=64, kernel_size=(3,3), padding="same")(input)
            # [nb_residual_unit] Residual Units
            residual_output = ResUnits(_residual_unit, nb_filter=64,
                              repetations=nb_residual_unit, bn=bn)(conv1)
            # Conv2
            if (bn2):
                residual_output = BatchNormalization(axis=1)(residual_output)
            activation = Activation('relu')(residual_output)
            conv2 = Conv2D(
                filters=nb_flow, kernel_size=(3,3), padding="same")(activation)
            outputs.append(conv2)

    # parameter-matrix-based fusion
    if len(outputs) == 1:
        main_output = outputs[0]
    else:
        from deepst.models.iLayer import iLayer
        new_outputs = []
        for output in outputs:
            print(output.shape)
            new_outputs.append(iLayer()(output))
        main_output = Add()(new_outputs)

    # fusing with external component
    if external_dim != None and external_dim > 0:
        # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(10)(external_input)
        embedding = Activation('relu')(embedding)
        h1 = Dense(nb_flow * map_height * map_width)(embedding)
        activation = Activation('relu')(h1)
        external_output = Reshape((nb_flow, map_height, map_width))(activation)
        main_output = Add()([main_output, external_output])
    else:
        print('external_dim:', external_dim)

    main_output = Activation('tanh')(main_output)
    model = Model(main_inputs, main_output)

    return model


if __name__ == '__main__':
    model = stresnet(external_dim=28, nb_residual_unit=12)
    #plot(model, to_file='ST-ResNet.png', show_shapes=True)
    model.summary()
