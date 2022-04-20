from keras.optimizers import Adam
from keras.models import Model
from keras.layers import (
    Input, Dense, Conv3D, Flatten, Dropout, MaxPooling3D,
    Activation, Lambda, Reshape, Concatenate, LSTM
)
from functools import reduce
from keras.backend import sigmoid
import numpy as np

import src.metrics as metrics

def swish(x, beta = 1):
    return (x * sigmoid(beta * x))

def CLoST3D(city, X_train, y_train,
            conv_filt = 64, kernel_sz = (2, 3, 3),
            mask = np.empty(0),
            lstm = None, lstm_number = 0,
            add_external_info = False):
  
    # Input:
    # - mask: np.array. Filter that is applied to the data output to the model. If not passed, no filter is applied
    # - lstm: int. Parameter to pass to the LSTM layer. If equal to None, the LSTM layer is not added.
    # - add_external_info: bool. Parameter to insert external information or not.

    X_train, ext_train = X_train # split flow volumes and ext features

    main_inputs = []

    start = Input(shape= (X_train.shape[1], X_train.shape[2] , X_train.shape[3], 2))

    main_inputs.append(start)
    main_output = main_inputs[0]

    x = Conv3D(conv_filt / 2, kernel_size = kernel_sz, activation='relu')(main_output)
    x = MaxPooling3D(pool_size=(1, 2, 2))(x)
    x = Dropout(0.25)(x)
    x = Conv3D(conv_filt, kernel_size = kernel_sz, activation='relu', padding = 'same')(x)
    x = MaxPooling3D(pool_size=(1, 2, 2))(x)
    if city == 'BJ':
      x = Dropout(0.25)(x)
      x = Conv3D(conv_filt, kernel_size = kernel_sz, activation='relu', padding = 'same')(x)
      x = MaxPooling3D(pool_size=(1, 2, 2))(x)
    x = Flatten()(x)
    x = Dense(128,  activation = 'relu') (x)
    if lstm != None:
      x = Reshape((x.shape[1], 1))(x)
      for num in range(lstm_number):
        if city == 'BJ':
          x = LSTM(int(lstm/(num+1)), return_sequences=True)(x)
        elif city == 'NY':
          x = LSTM(int(lstm), return_sequences=True)(x)
    x = Flatten()(x)
    if add_external_info:
      external_input = Input(shape=ext_train.shape[1:])
      main_inputs.append(external_input)
      x_ext = Dense(units=10, activation='relu')(external_input)
      x_ext = Dense(units=reduce(lambda e1, e2: e1*e2, y_train.shape[1:]), activation='relu')(x_ext)
      # x = Flatten()(x)
      x = Concatenate(axis = -1)([x, x_ext])
    x = Dense(reduce(lambda e1, e2: e1*e2, y_train.shape[1:]))(x)
    x = Reshape(y_train.shape[1:])(x)
    x = Activation(swish)(x)
    if mask.shape[0] != 0:
      x = Lambda(lambda el: el * mask)(x) 
    model = Model(main_inputs, x)
    return model

def build_model(
    city, X_train, y_train, conv_filt, kernel_sz, mask, lstm, lstm_number,
    add_external_info, lr=0.0001, save_model_pic=None):

    model = CLoST3D(city, X_train, y_train, conv_filt, kernel_sz, mask, lstm, lstm_number, add_external_info)
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    # model.summary()
    if (save_model_pic):
        from keras.utils.vis_utils import plot_model
        plot_model(model, to_file=f'{save_model_pic}.png', show_shapes=True)
    return model
