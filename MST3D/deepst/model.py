import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (
    Input,
    Conv3D,
    MaxPool3D,
    Dropout,
    Flatten,
    Activation,
    Add,
    Dense,
    Reshape,
    BatchNormalization
)
from keras.models import Model

# custom layer for branches fusion
class LinearLayer(tf.keras.layers.Layer):
  def __init__(self):
    super(LinearLayer, self).__init__()
    # self.num_outputs = num_outputs

  def build(self, input_shape):
    self.kernel1 = self.add_weight("kernel1", input_shape[0][1:])
    self.kernel2 = self.add_weight("kernel2", input_shape[1][1:])
    self.kernel3 = self.add_weight("kernel3", input_shape[2][1:])


  def call(self, inputs):
    return (
        tf.math.multiply(inputs[0], self.kernel1)
        + tf.math.multiply(inputs[1], self.kernel2)
        + tf.math.multiply(inputs[2], self.kernel3)
    )

'''
    MST3D implementation for BikeNYC and TaxiNYC
'''
def mst3d_nyc(len_c, len_p, len_t, nb_flow=2, map_height=16, map_width=8, external_dim=8):
    '''
    C - Temporal Closeness
    P - Period
    T - Trend
    external_dim
    '''

    # main input
    main_inputs = []
    outputs = []
    for len in [len_c, len_p, len_t]:
        if len is not None:
            input = Input(shape=(len, map_height, map_width, nb_flow))
            main_inputs.append(input)

            # the first convolutional layer has 32 filters and kernel size of (2,3,3)
            # set stride to (2,1,1) to reduce depth
            stride = (1,1,1)
            nb_filters = 32
            kernel_size = (2,3,3)

            conv1 = Conv3D(nb_filters, kernel_size, padding='same', activation='relu', strides=stride)(input)
            maxPool1 = MaxPool3D((1,2,2))(conv1)
            maxPool1 = BatchNormalization()(maxPool1)
            dropout1 = Dropout(0.25)(maxPool1)
            print(dropout1.shape)

            # the second layers have 64 filters
            nb_filters = 64

            conv2 = Conv3D(nb_filters, kernel_size, padding='same', activation='relu', strides=stride)(dropout1)
            maxPool2 = MaxPool3D((1,2,2))(conv2)
            maxPool2 = BatchNormalization()(maxPool2)
            dropout2 = Dropout(0.25)(maxPool2)
            print(dropout2.shape)

            outputs.append(dropout2)

    # parameter-matrix-based fusion
    fusion = LinearLayer()(outputs)
    flatten = Flatten()(fusion)

    # fusing with external component
    if external_dim != None and external_dim > 0:
        # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(10)(external_input)
        embedding = Activation('relu')(embedding)
        # h1 = Dense(nb_filters * 2 * map_height/4 * map_width/4)(embedding)
        h1 = Dense(flatten.shape[1])(embedding)
        activation = Activation('relu')(h1)
        main_output = Add()([flatten, activation])

    # reshape and tanh activation
    main_output = Dense(nb_flow * map_height * map_width)(main_output)
    main_output = Reshape((map_height, map_width, nb_flow))(main_output)
    main_output = Activation('tanh')(main_output)

    model = Model(main_inputs, main_output)

    return model

'''
    MST3D implementation for TaxiBJ
'''
def mst3d_bj(len_c, len_p, len_t, nb_flow=2, map_height=32, map_width=32, external_dim=8):
    '''
    C - Temporal Closeness
    P - Period
    T - Trend
    external_dim
    '''

    # main input
    main_inputs = []
    outputs = []
    for i, len in enumerate([len_c, len_p, len_t]):
        if len is not None:
            input = Input(shape=(len, map_height, map_width, nb_flow))
            main_inputs.append(input)

            # the first convolutional layer has 32 filters and kernel size of (2,3,3)
            # set stride to (2,1,1) to reduce depth
            stride = (1,1,1)
            nb_filters = 32
            conv1 = Conv3D(nb_filters, (2,3,3), padding='same', activation='relu', strides=stride)(input)
            maxPool1 = MaxPool3D((1,2,2))(conv1)
            maxPool1 = BatchNormalization()(maxPool1)
            dropout1 = Dropout(0.25)(maxPool1)
            # print(dropout1.shape[1])

            # the second and third layers have 64 filters. closeness branch has kernel
            # size of (2,3,3). the other branches have kernel size of (1,3,3).
            # Stride is (2,1,1) if previous layer has output depth>2
            nb_filters = 64
            if (i == 0):
              kernel_size = (2,3,3)
            else:
              kernel_size = (1,3,3)
            if (dropout1.shape[1] > 2):
              stride = (1,1,1)
            else:
              stride = (1,1,1)

            conv2 = Conv3D(nb_filters, kernel_size, padding='same', activation='relu', strides=stride)(dropout1)
            maxPool2 = MaxPool3D((1,2,2))(conv2)
            maxPool2 = BatchNormalization()(maxPool2)
            dropout2 = Dropout(0.25)(maxPool2)
            # print(dropout2.shape)

            if (dropout2.shape[1] > 2):
              stride = (1,1,1)
            else:
              stride = (1,1,1)

            conv3 = Conv3D(nb_filters, kernel_size, padding='same', activation='relu')(dropout2)
            maxPool3 = MaxPool3D((1,2,2))(conv3)
            maxPool3 = BatchNormalization()(maxPool3)
            dropout3 = Dropout(0.25)(maxPool3)

            outputs.append(dropout3)

    # parameter-matrix-based fusion
    fusion = LinearLayer()(outputs)
    flatten = Flatten()(fusion)

    # fusing with external component
    if external_dim != None and external_dim > 0:
        # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(10)(external_input)
        embedding = Activation('relu')(embedding)
        h1 = Dense(nb_filters * len_c * map_height/8 * map_width/8)(embedding)
        activation = Activation('relu')(h1)
        main_output = Add()([flatten, activation])

    # reshape and tanh activation
    main_output = Dense(nb_flow * map_height * map_width)(main_output)
    main_output = Reshape((map_height, map_width, nb_flow))(main_output)
    main_output = Activation('tanh')(main_output)

    model = Model(main_inputs, main_output)

    return model

'''
    MST3D bj con len_t = 2
'''
def mst3d_bj_2(len_c, len_p, len_t = 2, nb_flow=2, map_height=32, map_width=32, external_dim=8):
    '''
    C - Temporal Closeness
    P - Period
    T - Trend
    external_dim
    '''

    # main input
    main_inputs = []
    outputs = []
    for i, len in enumerate([len_c, len_p, len_t]):
        if len is not None:
            input = Input(shape=(len, map_height, map_width, nb_flow))
            main_inputs.append(input)

            # the first convolutional layer has 32 filters and kernel size of (2,3,3)
            # set stride to (2,1,1) to reduce depth
            stride = (1,1,1)
            nb_filters = 32
            conv1 = Conv3D(nb_filters, (2,3,3), padding='same', activation='relu', strides=stride)(input)
            if len == 2:
                conv1 = tf.keras.layers.UpSampling3D(size=(2,1,1))(conv1)
            maxPool1 = MaxPool3D((1,2,2))(conv1)
            maxPool1 = BatchNormalization()(maxPool1)
            dropout1 = Dropout(0.25)(maxPool1)
            #print(dropout1.shape[1])

            # the second and third layers have 64 filters. closeness branch has kernel
            # size of (2,3,3). the other branches have kernel size of (1,3,3).
            # Stride is (2,1,1) if previous layer has output depth>2
            nb_filters = 64
            if (i == 0):
              kernel_size = (2,3,3)
            else:
              kernel_size = (1,3,3)
            if (dropout1.shape[1] > 2):
              stride = (1,1,1)
            else:
              stride = (1,1,1)

            conv2 = Conv3D(nb_filters, kernel_size, padding='same', activation='relu', strides=stride)(dropout1)
            maxPool2 = MaxPool3D((1,2,2))(conv2)
            maxPool2 = BatchNormalization()(maxPool2)
            dropout2 = Dropout(0.25)(maxPool2)
            # print(dropout2.shape)

            if (dropout2.shape[1] > 2):
              stride = (1,1,1)
            else:
              stride = (1,1,1)

            conv3 = Conv3D(nb_filters, kernel_size, padding='same', activation='relu')(dropout2)
            maxPool3 = MaxPool3D((1,2,2))(conv3)
            maxPool3 = BatchNormalization()(maxPool3)
            dropout3 = Dropout(0.25)(maxPool3)

            outputs.append(dropout3)

    # parameter-matrix-based fusion
    fusion = LinearLayer()(outputs)
    flatten = Flatten()(fusion)

    # fusing with external component
    if external_dim != None and external_dim > 0:
        # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(10)(external_input)
        embedding = Activation('relu')(embedding)
        h1 = Dense(nb_filters * len_c * map_height/8 * map_width/8)(embedding)
        activation = Activation('relu')(h1)
        main_output = Add()([flatten, activation])

    # reshape and tanh activation
    main_output = Dense(nb_flow * map_height * map_width)(main_output)
    main_output = Reshape((map_height, map_width, nb_flow))(main_output)
    main_output = Activation('tanh')(main_output)

    model = Model(main_inputs, main_output)

    return model

'''
    MST3D implementation for BikeNYC and TaxiNYC with len_t=2
'''
def mst3d_nyc_2(len_c, len_p, len_t, nb_flow=2, map_height=16, map_width=8, external_dim=8):
    '''
    C - Temporal Closeness
    P - Period
    T - Trend
    external_dim
    '''

    # main input
    main_inputs = []
    outputs = []
    for len in [len_c, len_p, len_t]:
        if len is not None:
            input = Input(shape=(len, map_height, map_width, nb_flow))
            main_inputs.append(input)

            # the first convolutional layer has 32 filters and kernel size of (2,3,3)
            # set stride to (2,1,1) to reduce depth
            stride = (1,1,1)
            nb_filters = 32
            kernel_size = (2,3,3)

            conv1 = Conv3D(nb_filters, kernel_size, padding='same', activation='relu', strides=stride)(input)
            if len == 2:
                conv1 = tf.keras.layers.UpSampling3D(size=(2,1,1))(conv1)
            maxPool1 = MaxPool3D((1,2,2))(conv1)
            maxPool1 = BatchNormalization()(maxPool1)
            dropout1 = Dropout(0.25)(maxPool1)
            print(dropout1.shape)

            # the second layers have 64 filters
            nb_filters = 64

            conv2 = Conv3D(nb_filters, kernel_size, padding='same', activation='relu', strides=stride)(dropout1)
            maxPool2 = MaxPool3D((1,2,2))(conv2)
            maxPool2 = BatchNormalization()(maxPool2)
            dropout2 = Dropout(0.25)(maxPool2)
            print(dropout2.shape)

            outputs.append(dropout2)

    # parameter-matrix-based fusion
    fusion = LinearLayer()(outputs)
    flatten = Flatten()(fusion)

    # fusing with external component
    if external_dim != None and external_dim > 0:
        # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(10)(external_input)
        embedding = Activation('relu')(embedding)
        # h1 = Dense(nb_filters * 2 * map_height/4 * map_width/4)(embedding)
        h1 = Dense(flatten.shape[1])(embedding)
        activation = Activation('relu')(h1)
        main_output = Add()([flatten, activation])

    # reshape and tanh activation
    main_output = Dense(nb_flow * map_height * map_width)(main_output)
    main_output = Reshape((map_height, map_width, nb_flow))(main_output)
    main_output = Activation('tanh')(main_output)

    model = Model(main_inputs, main_output)

    return model
