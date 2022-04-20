import tensorflow as tf
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
    TimeDistributed,
    Conv2DTranspose,
    LSTM,
    Add,
    Layer
)
from keras.optimizers import Adam
import numpy as np

import src.metrics as metrics

# Definizione dell'Attention
tf.config.run_functions_eagerly(True)
def hw_flatten(x):
    return tf.reshape(x, shape=[tf.shape(x)[0], x.shape[1] * x.shape[2], x.shape[-1]])


def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2])
    return gap


def global_sum_pooling(x):
    gsp = tf.reduce_sum(x, axis=[1, 2])
    return gsp


def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.compat.v1.image.resize_nearest_neighbor(x, size=new_size)


def down_sample(x):
    return tf.nn.avg_pool2d(x, ksize=2, strides=2, padding='SAME')


def max_pooling(x):
    return tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME')


def conv(x, channels, kernel, stride):
    x = Conv2D(filters=channels, kernel_size=kernel, strides=stride)(x)
    return x


def attention(x, ch):
    batch_size, height, width, num_channels = tf.shape(x)[0], x.shape[1], x.shape[2], x.shape[3]
    f = conv(x, ch // 8, kernel=1, stride=1)  # [bs, h, w, c']
    g = conv(x, ch // 8, kernel=1, stride=1)  # [bs, h, w, c']
    h = conv(x, ch, kernel=1, stride=1)  # [bs, h, w, c]

    # N = h * w
    s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

    beta = tf.nn.softmax(s)  # attention map

    o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
    gamma = tf.Variable(tf.zeros([1], dtype=tf.dtypes.float32), name="gamma", trainable=True)

    o = tf.reshape(o, shape=[batch_size, height, width, num_channels])  # [bs, h, w, C]
    x = gamma * o + x

    return x


class Attention_2(Layer):
    def __init__(self, ch):
        super(Attention_2, self).__init__()
        self.ch = ch
    
    def build(self, input_shape):
        super(Attention_2, self).build(input_shape)

    def call(self, x):
        batch_size, height, width, num_channels = tf.shape(x)[0], x.shape[1], x.shape[2], x.shape[3]
        f = conv(x, self.ch // 8, kernel=1, stride=1)  # [bs, h, w, c']
        f = max_pooling(
            f)  # estrapola solo i valori più grandi, avendo poi padding same allora non c'è una riduzione della dimensionalità

        g = conv(x, self.ch // 8, kernel=1, stride=1)  # [bs, h, w, c']

        h = conv(x, self.ch // 2, kernel=1, stride=1)  # [bs, h, w, c]
        h = max_pooling(h)

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s)  # attention map

        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
        gamma = tf.Variable(lambda: tf.zeros([1], dtype=tf.dtypes.float32), trainable=True, name="gamma", shape=tf.TensorShape([1]))

        o = tf.reshape(o, shape=[batch_size, height, width, num_channels // 2])  # [bs, h, w, C]
        o = conv(o, self.ch, kernel=1, stride=1)
        x = gamma * o + x

        return x

    def compute_output_shape(self, input_shape):
        return (input_shape)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'ch': self.ch
        })
        return config

class MultiplicativeUnit():
    """Initialize the multiplicative unit.
    Args:
       layer_name: layer names for different multiplicative units.
       filter_size: int tuple of the height and width of the filter.
       num_hidden: number of units in output tensor.
    """
    def __init__(self, layer_name, num_hidden, filter_size):
        self.layer_name = layer_name
        self.num_features = num_hidden
        self.filter_size = filter_size

    def __call__(self, h, reuse=False):
        with tf.compat.v1.variable_scope(self.layer_name, reuse=reuse):
            g1 = Conv2D(
                self.num_features, self.filter_size, padding='same', activation=tf.sigmoid,
                # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                )(h)
            g2 = Conv2D(
                self.num_features, self.filter_size, padding='same', activation=tf.sigmoid,
                # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                )(h)
            g3 = Conv2D(
                self.num_features, self.filter_size, padding='same', activation=tf.sigmoid,
                # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                )(h)
            u = Conv2D(
                self.num_features, self.filter_size, padding='same', activation=tf.tanh,
                # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                )(h)
            g2_h = tf.multiply(g2, h)
            g3_u = tf.multiply(g3, u)
            mu = tf.multiply(g1, tf.tanh(g2_h + g3_u))
            return mu

class CMU():
    """Initialize the causal multiplicative unit.
    Args:
       layer_name: layer names for different causal multiplicative unit.
       filter_size: int tuple of the height and width of the filter.
       num_hidden: number of units in output tensor.
    """
    def __init__(self, layer_name, num_hidden, filter_size):
        self.layer_name = layer_name
        self.num_features = num_hidden
        self.filter_size = filter_size

    def __call__(self, h1, h2, stride=False, reuse=False):
        with tf.compat.v1.variable_scope(self.layer_name, reuse=reuse):
            hl = MultiplicativeUnit('multiplicative_unit_1', self.num_features, self.filter_size)(h1, reuse=reuse)
            if not stride:
                hl = MultiplicativeUnit('multiplicative_unit_1', self.num_features, self.filter_size)(hl, reuse=True)
            hr = MultiplicativeUnit('multiplicative_unit_2', self.num_features, self.filter_size)(h2, reuse=reuse)
            h = tf.add(hl, hr)
            h_sig = Conv2D(
                self.num_features, self.filter_size, padding='same', activation=tf.sigmoid,
                # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                )(h)
            h_tan = Conv2D(
                self.num_features, self.filter_size, padding='same', activation=tf.tanh,
                # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                )(h)
            h = tf.multiply(h_sig, h_tan)
            return h

def predcnn_perframe(xs, num_hidden, filter_size, input_length, reuse):
    with tf.compat.v1.variable_scope('frame_prediction', reuse=reuse):
        for i in range(input_length-1):
            temp = []
            for j in range(input_length-i-1):
                h1 = xs[j]
                h2 = xs[j+1]
                h = CMU('causal_multiplicative_unit_'+str(i+1), num_hidden, filter_size)(h1, h2, stride=False, reuse=bool(temp))
                temp.append(h)
            xs = temp
        return xs[0]

def my_conv(input_layer, filters, activation, kernel_size=3, time_distributed=False):
    if (time_distributed):
        l = TimeDistributed(Conv2D(filters, kernel_size, padding='same', activation=activation))(input_layer)
        l = TimeDistributed(BatchNormalization())(l)
        l = TimeDistributed(Attention_2(filters))(l)
        return l
    else:
        l = Conv2D(filters, kernel_size, padding='same', activation=activation)(input_layer)
        l = BatchNormalization()(l)
        return l

def my_downsampling(input_layer, filters):
    l = TimeDistributed(Conv2D(filters, (2,2), (2,2), activation='relu'))(input_layer)
    l = TimeDistributed(BatchNormalization())(l)
    return l

def my_conv_transpose(input_layer, skip_connection_layer):
    l = Conv2DTranspose(input_layer.shape[-1], (2,2), (2,2))(input_layer)
    l = Attention_2(input_layer.shape[-1])(l)
    skl = skip_connection_layer[:,-1,:,:,:] # extract features from last image
    l = Add()([l, skl])
    l = Activation('relu')(l)
    l = BatchNormalization()(l)
    return l


def my_model(len_c, len_p, len_t, nb_flow=2, map_height=32, map_width=32,
             external_dim=8, encoder_blocks=3, filters=[32,64,64,16], kernel_size=3,
             lstm_units=16):

    main_inputs = []
    #ENCODER
    # input layer tx32x32x2
    t = len_c+len_p*2+len_t*2
    input = Input(shape=((t, map_height, map_width, nb_flow)))
    main_inputs.append(input)
    x = input

    # build encoder blocks
    skip_connection_layers = []
    for i in range(0, encoder_blocks):        
        # conv + relu + bn
        x = my_conv(x, filters[i], 'relu', kernel_size, time_distributed=True)
        # append layer to skip connection list
        skip_connection_layers.append(x)
        # max pool
        x = my_downsampling(x, x.shape[-1])

    # last convolution tx4x4x16
    x = my_conv(x, filters[-1], 'relu', kernel_size)
    s = x.shape
    # print(s)

    list_features = [x[:,i,:,:,:] for i in range(x.shape[1])]
    x = predcnn_perframe(list_features, s[-1], kernel_size, t, reuse=False)
    # x = TimeDistributed(Flatten())(x)
    # units = x.shape[-1]
    # x = LSTM(lstm_units, return_sequences=True)(x)
    # x = LSTM(lstm_units, return_sequences=True)(x)
    # x = LSTM(lstm_units, return_sequences=True)(x)
    # x = LSTM(units, return_sequences=False)(x)
    x = Reshape((s[2:]))(x)

    # merge external features
    if external_dim != None and external_dim > 0:
        # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(units=10, activation='relu')(external_input)
        h1 = Dense(units=s[2]*s[3]*s[4], activation='relu')(embedding)
        external_output = Reshape((s[2], s[3], s[4]))(h1)
        x = Add()([x, external_output])

    # build decoder blocks
    for i in reversed(range(0, encoder_blocks)):
        # conv + relu + bn
        x = my_conv(x, filters[i], 'relu', kernel_size)
        print(x.shape)
        # conv_transpose + skip_conn + relu + bn
        x = my_conv_transpose(x, skip_connection_layers[i])

    # last convolution + tanh + bn 32x32x2
    output = my_conv(x, nb_flow, 'tanh')

    return Model(main_inputs, output)

def build_model(len_c, len_p, len_t, nb_flow=2, map_height=32, map_width=32,
                external_dim=8, encoder_blocks=3, filters=[32,64,64,16],
                kernel_size=3, lstm_units=16, lr=0.0001, save_model_pic=None):
    model = my_model(len_c, len_p, len_t, nb_flow, map_height, map_width,
                     external_dim, encoder_blocks, filters, kernel_size, lstm_units)
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    # model.summary()
    if (save_model_pic):
        from keras.utils.vis_utils import plot_model
        plot_model(model, to_file=f'{save_model_pic}.png', show_shapes=True)
    return model