from ST3DNet import *
import pickle
from utils import *
import os
import h5py
import math
import numpy as np
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from evaluation import evaluate

def save_to_csv(score, csv_name):
    if not os.path.isfile(csv_name):
        if os.path.isdir('results') is False:
            os.mkdir('results')
        with open(csv_name, 'a', encoding = "utf-8") as file:
            file.write(
                    'rsme_in,rsme_out,rsme_tot,'
                    'mape_in,mape_out,mape_tot,'
                    'ape_in,ape_out,ape_tot'
                    )
            file.write("\n")
            file.close()
    with open(csv_name, 'a', encoding = "utf-8") as file:
        file.write(f'{score[0]},{score[1]},{score[2]},{score[3]},'
                f'{score[4]},{score[5]},{score[6]},{score[7]},{score[8]}'
                )
        file.write("\n")
        file.close()

# tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:  # Currently, memory growth needs to be the same across GPUs
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)  # Memory growth must be set before GPUs have been initialized

### 32x32
# params
T = 24*2  # number of time intervals in one day
lr = 0.0001  # learning rate
# lr = 0.00002  # learning rate
len_closeness = 6  # length of closeness dependent sequence
len_period = 0  # length of peroid dependent sequence
len_trend = 2  # length of trend dependent sequence
nb_residual_unit = 7   # number of residual units
nb_flow = 2  # there are two types of flows: new-flow and end-flow
days_test = 7  
len_test = T * days_test
map_height, map_width = 32, 32  # grid size
m_factor = 1

# load data
filename = os.path.join("../data", 'CACHE', 'ST3DNet', 'Rome_c%d_p%d_t%d_noext'%(len_closeness, len_period, len_trend))
f = open(filename, 'rb')
X_train = pickle.load(f)
Y_train = pickle.load(f)
X_test = pickle.load(f)
Y_test = pickle.load(f)
mmn = pickle.load(f)
external_dim = pickle.load(f)
timestamp_train = pickle.load(f)
timestamp_test = pickle.load(f)

for i in X_train:
    print(i.shape)

Y_train = mmn.inverse_transform(Y_train)  # X is MaxMinNormalized, Y is real value
Y_test = mmn.inverse_transform(Y_test)

c_conf = (len_closeness, nb_flow, map_height,
              map_width) if len_closeness > 0 else None
t_conf = (len_trend, nb_flow, map_height,
          map_width) if len_trend > 0 else None

# build model
model = ST3DNet(c_conf=c_conf, t_conf=t_conf, external_dim=external_dim, nb_residual_unit=nb_residual_unit)
adam = Adam(lr=lr)
model.compile(loss='mse', optimizer=adam, metrics=[rmse])

## single-step-prediction no TL
nb_epoch = 200
batch_size = 16
hyperparams_name = 'st3dnet_roma32x32'
fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))
model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')
history = model.fit(X_train, Y_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    validation_data=(X_test, Y_test),
                    callbacks=[model_checkpoint],
                    verbose=2)

# predict
Y_pred = model.predict(X_test)  # compute predictions

# evaluate
score = evaluate(Y_test, Y_pred)  # evaluate performance

# save to csv
csv_name = os.path.join('results', f'roma32x32_results.csv')
save_to_csv(score, csv_name)

## TL without re-training
# load weights
model_fname = 'TaxiBJ.c6.p0.t2.resunit7.lr0.0001.cont.noMeteo.best.h5'
model.load_weights(os.path.join('../best_models', 'ST3DNet', model_fname))

# predict
Y_pred = model.predict(X_test)  # compute predictions

# evaluate
score = evaluate(Y_test, Y_pred)  # evaluate performance

# save to csv
csv_name = os.path.join('results', f'TL_taxiBJ_roma32x32_results.csv')
save_to_csv(score, csv_name)

# save real vs predicted
fname = 'st3dnet_RomaNord32x32.h5'
h5 = h5py.File(fname, 'w')
h5.create_dataset('Y_real', data=Y_test)
h5.create_dataset('Y_pred', data=Y_pred)
h5.create_dataset('timestamps', data=timestamp_test)
h5.create_dataset('max', data=mmn._max)
h5.close()

## TL with re-training
nb_epoch = 200
batch_size = 16
hyperparams_name = 'TaxiBJ_Rome'
fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))
model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')
history = model.fit(X_train, Y_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    validation_data=(X_test, Y_test),
                    callbacks=[model_checkpoint],
                    verbose=2)

# evaluate after training
model.load_weights(fname_param)
Y_pred = model.predict(X_test)  # compute predictions

# evaluate
score = evaluate(Y_test, Y_pred)  # evaluate performance

# save to csv
csv_name = os.path.join('results', f'TL_taxiBJ_roma32x32_training_results.csv')
save_to_csv(score, csv_name)

# save real vs predicted
fname = 'st3dnet_RomaNord32x32_trained.h5'
h5 = h5py.File(fname, 'w')
h5.create_dataset('Y_real', data=Y_test)
h5.create_dataset('Y_pred', data=Y_pred)
h5.create_dataset('timestamps', data=timestamp_test)
h5.create_dataset('max', data=mmn._max)
h5.close()


### 16x8
# params
T = 24
lr = 0.0001  # learning rate
# lr = 0.00002  # learning rate
len_closeness = 6  # length of closeness dependent sequence
len_period = 0  # length of peroid dependent sequence
len_trend = 2  # length of trend dependent sequence
nb_residual_unit = 5   # number of residual units
nb_flow = 2  # there are two types of flows: new-flow and end-flow
days_test = 7  
len_test = T * days_test
map_height, map_width = 16, 8  # grid size
m_factor = 1

# load data
filename = os.path.join("../data", 'CACHE', 'ST3DNet', 'Rome16x8_c%d_p%d_t%d_noext'%(len_closeness, len_period, len_trend))
f = open(filename, 'rb')
X_train = pickle.load(f)
Y_train = pickle.load(f)
X_test = pickle.load(f)
Y_test = pickle.load(f)
mmn = pickle.load(f)
external_dim = pickle.load(f)
timestamp_train = pickle.load(f)
timestamp_test = pickle.load(f)

for i in X_train:
    print(i.shape)

Y_train = mmn.inverse_transform(Y_train)  # X is MaxMinNormalized, Y is real value
Y_test = mmn.inverse_transform(Y_test)

c_conf = (len_closeness, nb_flow, map_height,
              map_width) if len_closeness > 0 else None
t_conf = (len_trend, nb_flow, map_height,
          map_width) if len_trend > 0 else None

# build model
model = ST3DNet(c_conf=c_conf, t_conf=t_conf, external_dim=external_dim, nb_residual_unit=nb_residual_unit)
adam = Adam(lr=lr)
model.compile(loss='mse', optimizer=adam, metrics=[rmse])

## single-step-prediction no TL
nb_epoch = 200
batch_size = 16
hyperparams_name = 'st3dnet_roma16x8'
fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))
model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')
history = model.fit(X_train, Y_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    validation_data=(X_test, Y_test),
                    callbacks=[model_checkpoint],
                    verbose=2)

# predict
Y_pred = model.predict(X_test)  # compute predictions

# evaluate
score = evaluate(Y_test, Y_pred)  # evaluate performance

# save to csv
csv_name = os.path.join('results', f'roma16x8_results.csv')
save_to_csv(score, csv_name)


## TL without re-training
# load weights
model_fname = 'TaxiNYC2.c6.p0.t2.resunits_5.lr_0.00095.batchsize_16.best.h5'
model.load_weights(os.path.join('../best_models', 'ST3DNet', model_fname))

# predict
Y_pred = model.predict(X_test)  # compute predictions

# evaluate
score = evaluate(Y_test, Y_pred)  # evaluate performance

# save to csv
csv_name = os.path.join('results', f'TL_taxiNY_roma16x8_results.csv')
save_to_csv(score, csv_name)

# save real vs predicted
fname = 'st3dnet_RomaNord16x8.h5'
h5 = h5py.File(fname, 'w')
h5.create_dataset('Y_real', data=Y_test)
h5.create_dataset('Y_pred', data=Y_pred)
h5.create_dataset('timestamps', data=timestamp_test)
h5.create_dataset('max', data=mmn._max)
h5.close()

## TL with re-training
nb_epoch = 200
batch_size = 16
hyperparams_name = 'TaxiNY_Rome'
fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))
model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')
history = model.fit(X_train, Y_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    validation_data=(X_test, Y_test),
                    callbacks=[model_checkpoint],
                    verbose=2)

# evaluate after training
model.load_weights(fname_param)
Y_pred = model.predict(X_test)  # compute predictions

# evaluate
score = evaluate(Y_test, Y_pred)  # evaluate performance

# save to csv
csv_name = os.path.join('results', f'TL_taxiNY_roma16x8_training_results.csv')
save_to_csv(score, csv_name)

# save real vs predicted
fname = 'st3dnet_RomaNord16x8_trained.h5'
h5 = h5py.File(fname, 'w')
h5.create_dataset('Y_real', data=Y_test)
h5.create_dataset('Y_pred', data=Y_pred)
h5.create_dataset('timestamps', data=timestamp_test)
h5.create_dataset('max', data=mmn._max)
h5.close()
