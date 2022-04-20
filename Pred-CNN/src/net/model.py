import tensorflow as tf
from keras.layers import Input, Conv2D
from keras.models import Model
from keras.optimizers import Adam

from src.layers.ResidualMultiplicativeBlock import ResidualMultiplicativeBlock as rmb
from src.layers.CascadeMultiplicativeUnit import CascadeMultiplicativeUnit as cmu
import src.metrics as metrics


# def predcnn(params, mask_true, num_hidden, filter_size, seq_length=20, input_length=10):
def predcnn(input_length, map_height, map_width, channels=2, encoder_length=2,
            decoder_length=3, num_hidden=64, filter_size=(3,3),ext_dim=28):

    seq_length = input_length + 1 # next frame prediction

    with tf.compat.v1.variable_scope('predcnn'):
        # encoder
        main_input = Input(shape=(input_length, map_height, map_width, channels))
        ext_input = Input(shape=(ext_dim))
        encoder_output = []
        for i in range(input_length):
            reuse = bool(encoder_output)
            ims = main_input[:,i]
            input = resolution_preserving_cnn_encoders(ims, num_hidden, filter_size, encoder_length, reuse)
            encoder_output.append(input)

        # predcnn & decoder
        output = []
        for i in range(seq_length - input_length):
            reuse = bool(output)
            out = predcnn_perframe(encoder_output[i:i+input_length], num_hidden, filter_size, input_length, reuse)
            out = cnn_docoders(out, num_hidden, filter_size, channels, decoder_length, reuse)
            output.append(out)

            # ims = mask_true[:, 0] * images[:, input_length + i] + (1 - mask_true[:, 0]) * out
            # input = resolution_preserving_cnn_encoders(ims, num_hidden, filter_size, encoder_length, reuse=True)
            # encoder_output.append(input)

    # # transpose output and compute loss
    # gen_images = tf.stack(output)
    # # [batch_size, seq_length, height, width, channels]
    # gen_images = tf.transpose(gen_images, [1, 0, 2, 3, 4])
    # loss = tf.nn.l2_loss(gen_images - images[:, input_length:])

    # return [gen_images, loss]
    return Model([main_input,ext_input], output)


def resolution_preserving_cnn_encoders(x, num_hidden, filter_size, encoder_length, reuse):
    with tf.compat.v1.variable_scope('resolution_preserving_cnn_encoders', reuse=reuse):
        x = Conv2D(num_hidden, filter_size, padding='same', activation=None,
                   kernel_initializer=tf.keras.initializers.GlorotUniform(),
                  #  name='input_conv'
                  )(x)
        for i in range(encoder_length):
            x = rmb('residual_multiplicative_block_'+str(i+1), num_hidden // 2, filter_size)(x)
        return x


def predcnn_perframe(xs, num_hidden, filter_size, input_length, reuse):
    with tf.compat.v1.variable_scope('frame_prediction', reuse=reuse):
        for i in range(input_length-1):
            temp = []
            for j in range(input_length-i-1):
                h1 = xs[j]
                h2 = xs[j+1]
                h = cmu('causal_multiplicative_unit_'+str(i+1), num_hidden, filter_size)(h1, h2, stride=False, reuse=bool(temp))
                temp.append(h)
            xs = temp
        return xs[0]


def cnn_docoders(x, num_hidden, filter_size, output_channels, decoder_length, reuse):
    with tf.compat.v1.variable_scope('cnn_decoders', reuse=reuse):
        for i in range(decoder_length):
            x = rmb('residual_multiplicative_block_'+str(i+1), num_hidden // 2, filter_size)(x)
        x = Conv2D(output_channels, filter_size, padding='same',
                   kernel_initializer=tf.keras.initializers.GlorotUniform(),
                  #  name='output_conv'
                   )(x)
        return x

def custom_loss(y_true, y_pred):
    return tf.nn.l2_loss(y_true - y_pred)

def build_model(input_length, map_height, map_width, channels=2, encoder_length=2,
                decoder_length=3, num_hidden=64, filter_size=(3,3),lr=0.0001,
                save_model_pic=None):

    model = predcnn(input_length, map_height, map_width, channels, encoder_length,
            decoder_length, num_hidden, filter_size)
    adam = Adam(lr=lr)
    model.compile(loss=custom_loss, optimizer=adam, metrics=[metrics.rmse])
    # model.summary()
    if (save_model_pic):
        from keras.utils.vis_utils import plot_model
        plot_model(model, to_file=f'{save_model_pic}.png', show_shapes=True)
    return model
