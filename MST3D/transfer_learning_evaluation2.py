from __future__ import print_function
import os
import sys
import pickle
import time
import numpy as np
import h5py

import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

import deepst.metrics as metrics
from deepst.datasets import carRome2
from deepst.model import mst3d_bj_2, mst3d_nyc_2
from deepst.evaluation import evaluate


def save_to_csv(score, csv_name):
    if not os.path.isfile(csv_name):
        if os.path.isdir('results_roma_bergamo') is False:
            os.mkdir('results_roma_bergamo')
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



path_model = 'MODEL_ROMA_BERGAMO'
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)

path_confronto = 'Confronto'
if os.path.isdir(path_confronto) is False:
    os.mkdir(path_confronto)

def build_model_bj(save_model_pic=False):
    model = mst3d_bj_2(len_closeness, len_period, len_trend, nb_flow, map_height, map_width, external_dim)
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    # model.summary()
    if (save_model_pic):
        from keras.utils.vis_utils import plot_model
        plot_model(model, to_file='TaxiBJ_model.png', show_shapes=True)
    return model

def build_model_ny(len_c, len_p, len_t, nb_flow, map_height, map_width,
                external_dim, save_model_pic=False, lr=0.00015):
    model = mst3d_nyc_2(
      len_c, len_p, len_t,
      nb_flow, map_height, map_width,
      external_dim
    )
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    # model.summary()
    if (save_model_pic):
        from keras.utils.vis_utils import plot_model
        plot_model(model, to_file='TaxiNYC_model.png', show_shapes=True)

    return model

def read_cache(fname, preprocess_name):
    mmn = pickle.load(open(preprocess_name, 'rb'))

    f = h5py.File(fname, 'r')
    num = int(f['num'].value)
    X_train, Y_train, X_test, Y_test = [], [], [], []
    for i in range(num):
        X_train.append(f['X_train_%i' % i].value)
        X_test.append(f['X_test_%i' % i].value)
    Y_train = f['Y_train'].value
    Y_test = f['Y_test'].value
    external_dim = f['external_dim'].value
    timestamp_train = f['T_train'].value
    timestamp_test = f['T_test'].value
    f.close()

    return X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test

def cache(fname, X_train, Y_train, X_test, Y_test, external_dim, timestamp_train, timestamp_test):
    h5 = h5py.File(fname, 'w')
    h5.create_dataset('num', data=len(X_train))

    for i, data in enumerate(X_train):
        h5.create_dataset('X_train_%i' % i, data=data)
    # for i, data in enumerate(Y_train):
    for i, data in enumerate(X_test):
        h5.create_dataset('X_test_%i' % i, data=data)
    h5.create_dataset('Y_train', data=Y_train)
    h5.create_dataset('Y_test', data=Y_test)
    external_dim = -1 if external_dim is None else int(external_dim)
    h5.create_dataset('external_dim', data=external_dim)
    h5.create_dataset('T_train', data=timestamp_train)
    h5.create_dataset('T_test', data=timestamp_test)
    h5.close()

### 32x32
# parameters
DATAPATH = '../data'
CACHEDATA = True  # cache data or NOT
path_cache = os.path.join(DATAPATH, 'CACHE', 'MST3D')  # cache path
# nb_epoch = 100  # number of epoch at training stage
# nb_epoch_cont = 100  # number of epoch at training (cont) stage
# batch_size = 64  # batch size
T = 24*2  # number of time intervals in one day
lr = 0.0001  # learning rate
len_closeness = 4  # length of closeness dependent sequence - should be 6
len_period = 4  # length of peroid dependent sequence
len_trend = 2  # length of trend dependent sequence

nb_flow = 2  # there are two types of flows: inflow and outflow

days_test = 7 
len_test = T * days_test
map_height, map_width = 32, 32  # grid size

if CACHEDATA and os.path.isdir(path_cache) is False:
    os.mkdir(path_cache)

# load data
print("loading data...")
preprocess_name = 'preprocessing_rome_2.pkl'
ts = time.time()
fname = os.path.join(path_cache, 'Rome_C{}_P{}_T{}_2.h5'.format(
    len_closeness, len_period, len_trend))
if os.path.exists(fname) and CACHEDATA:
    X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = read_cache(
        fname, preprocess_name)
    print("load %s successfully" % fname)
else:
    X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = carRome2.load_data(
        T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,
        preprocess_name=preprocess_name, meta_data=True, meteorol_data=False, holiday_data=True, datapath=DATAPATH)
    if CACHEDATA:
        cache(fname, X_train, Y_train, X_test, Y_test,
              external_dim, timestamp_train, timestamp_test)

print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])
print("\nelapsed time (loading data): %.3f seconds\n" % (time.time() - ts))

print('=' * 10)

# build model
model = build_model_bj(save_model_pic=False)

## single-step-prediction no TL
nb_epoch = 100
batch_size = 16
hyperparams_name = 'mst3d_roma32x32'
fname_param = os.path.join('MODEL_ROMA_BERGAMO', '{}.best.h5'.format(hyperparams_name))
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
score = evaluate(Y_test, Y_pred, mmn)  # evaluate performance

# save to csv
csv_name = os.path.join('results_roma_bergamo', f'roma32x32_results.csv')
save_to_csv(score, csv_name)

## TL without re-training
# load weights
model_fname = 'TaxiBJ.c4.p4.t2.iter6.best.h5'
model.load_weights(os.path.join('../best_models', 'MST3D', model_fname))

# predict
Y_pred = model.predict(X_test)  # compute predictions

# evaluate
score = evaluate(Y_test, Y_pred, mmn)  # evaluate performance

# save to csv
csv_name = os.path.join('results_roma_bergamo', f'TL_taxiBJ_roma32x32_results.csv')
save_to_csv(score, csv_name)

# save real vs predicted
fname = 'mst3d_RomaNord32x32.h5'
h5 = h5py.File(os.path.join(path_confronto,fname), 'w')
h5.create_dataset('Y_real', data=Y_test)
h5.create_dataset('Y_pred', data=Y_pred)
h5.create_dataset('timestamps', data=timestamp_test)
h5.create_dataset('max', data=mmn._max)
h5.close()

## TL with re-training
nb_epoch = 100
batch_size = 16
hyperparams_name = 'TaxiBJ_Rome'
fname_param = os.path.join('MODEL_ROMA_BERGAMO', '{}.best.h5'.format(hyperparams_name))
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
score = evaluate(Y_test, Y_pred, mmn)  # evaluate performance

# save to csv
csv_name = os.path.join('results_roma_bergamo', f'TL_taxiBJ_roma32x32_training_results.csv')
save_to_csv(score, csv_name)

# save real vs predicted
fname = 'mst3d_RomaNord32x32_trained.h5'
h5 = h5py.File(os.path.join(path_confronto,fname), 'w')
h5.create_dataset('Y_real', data=Y_test)
h5.create_dataset('Y_pred', data=Y_pred)
h5.create_dataset('timestamps', data=timestamp_test)
h5.create_dataset('max', data=mmn._max)
h5.close()


### 16x8
# parameters
DATAPATH = '../data'
CACHEDATA = True  # cache data or NOT
path_cache = os.path.join(DATAPATH, 'CACHE', 'MST3D')  # cache path
# nb_epoch = 100  # number of epoch at training stage
# nb_epoch_cont = 100  # number of epoch at training (cont) stage
# batch_size = 16  # batch size
T = 24  # number of time intervals in one day
lr = 0.0001  # learning rate
len_closeness = len_c = 4  # length of closeness dependent sequence - should be 6
len_period = len_p = 4  # length of peroid dependent sequence
len_trend = len_t = 2  # length of trend dependent sequence

nb_flow = 2  # there are two types of flows: inflow and outflow

days_test = 7 
len_test = T * days_test
map_height, map_width = 16, 8  # grid size

# load data
print("loading data...")
preprocess_name = 'preprocessing_rome16x8_2.pkl'
fname = os.path.join(path_cache, 'Rome16x8_C{}_P{}_T{}_2.h5'.format(
    len_c, len_p, len_t))
if os.path.exists(fname) and CACHEDATA:
    X_train, Y_train, \
    X_test, Y_test, mmn, external_dim, \
    timestamp_train, timestamp_test = read_cache(
        fname, preprocess_name)
    print("load %s successfully" % fname)
else:
    X_train, Y_train, \
    X_test, Y_test, mmn, external_dim, \
    timestamp_train, timestamp_test = carRome2.load_data(
        T=T, nb_flow=nb_flow, len_closeness=len_c, len_period=len_p, len_trend=len_t, len_test=len_test,
        preprocess_name=preprocess_name, meta_data=True,
        meteorol_data=True, holiday_data=True, datapath=DATAPATH, shape=(16,8))
    if CACHEDATA:
        cache(fname, X_train, Y_train, X_test, Y_test,
                external_dim, timestamp_train, timestamp_test)

print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])
print('=' * 10)

# build model
model = build_model_ny(
    len_c, len_p, len_t, nb_flow, map_height,
    map_width, external_dim,
    save_model_pic=False,
    lr=lr
)

## single-step-prediction no TL
nb_epoch = 100
batch_size = 16
hyperparams_name = 'mst3d_roma16x8'
fname_param = os.path.join('MODEL_ROMA_BERGAMO', '{}.best.h5'.format(hyperparams_name))
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
score = evaluate(Y_test, Y_pred, mmn)  # evaluate performance

# save to csv
csv_name = os.path.join('results_roma_bergamo', f'roma16x8_results.csv')
save_to_csv(score, csv_name)

## TL without re-training
# load weights
model_fname = 'TaxiNYC1.c4.p4.t2.lr_0.00034.batchsize_16.best.h5'
model.load_weights(os.path.join('../best_models', 'MST3D', model_fname))

# predict
Y_pred = model.predict(X_test)  # compute predictions

# evaluate
score = evaluate(Y_test, Y_pred, mmn)  # evaluate performance

# save to csv
csv_name = os.path.join('results_roma_bergamo', f'TL_taxiNY_roma16x8_results.csv')
save_to_csv(score, csv_name)

# save real vs predicted
fname = 'mst3d_RomaNord16x8.h5'
h5 = h5py.File(os.path.join(path_confronto,fname), 'w')
h5.create_dataset('Y_real', data=Y_test)
h5.create_dataset('Y_pred', data=Y_pred)
h5.create_dataset('timestamps', data=timestamp_test)
h5.create_dataset('max', data=mmn._max)
h5.close()

## TL with re-training
nb_epoch = 100
batch_size = 16
hyperparams_name = 'TaxiNY_Rome'
fname_param = os.path.join('MODEL_ROMA_BERGAMO', '{}.best.h5'.format(hyperparams_name))
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
score = evaluate(Y_test, Y_pred, mmn)  # evaluate performance

# save to csv
csv_name = os.path.join('results_roma_bergamo', f'TL_taxiNY_roma16x8_training_results.csv')
save_to_csv(score, csv_name)

# save real vs predicted
fname = 'model3_RomaNord16x8_trained.h5'
h5 = h5py.File(os.path.join(path_confronto,fname), 'w')
h5.create_dataset('Y_real', data=Y_test)
h5.create_dataset('Y_pred', data=Y_pred)
h5.create_dataset('timestamps', data=timestamp_test)
h5.create_dataset('max', data=mmn._max)
h5.close()
