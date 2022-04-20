from __future__ import print_function
import os
import _pickle as pickle
import numpy as np
import math
import h5py

import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from src.model import build_model
import src.metrics as metrics
from src.datasets import carRome2
from src.evaluation import evaluate
from cache_utils import cache, read_cache


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
### 32x32
# parameters
DATAPATH = '../data'
nb_epoch = 100  # number of epoch at training stage
batch_size = 64  # batch size
T = 24*2  # number of time intervals in one day
CACHEDATA = True  # cache data or NOT

lr = 0.0001  # learning rate
len_c = 4  # length of closeness dependent sequence
len_p = 1  # length of peroid dependent sequence
len_t = 0  # length of trend dependent sequence

nb_flow = 2  # there are two types of flows: new-flow and end-flow
# divide data into two subsets: Train & Test
days_test = 7
len_test = T*days_test
len_val = len_test # no val

map_height, map_width = 32, 32  # grid size

path_cache = os.path.join(DATAPATH, 'CACHE', '3D-CLoST')  # cache path
if CACHEDATA and os.path.isdir(path_cache) is False:
    os.mkdir(path_cache)

# load dataadd_half
print("loading data...")
preprocess_name = 'preprocessing_rome_2.pkl'
fname = os.path.join(path_cache, 'Rome_C{}_P{}_T{}_2.h5'.format(
    len_c, len_p, len_t))
if os.path.exists(fname) and CACHEDATA:
    X_train_all, Y_train_all, X_train, Y_train, \
    X_val, Y_val, X_test, Y_test, mmn, external_dim, \
    timestamp_train_all, timestamp_train, timestamp_val, timestamp_test, mask = read_cache(
        fname, preprocess_name)
    print("load %s successfully" % fname)
else:
    X_train_all, Y_train_all, X_train, Y_train, \
    X_val, Y_val, X_test, Y_test, mmn, external_dim, \
    timestamp_train_all, timestamp_train, timestamp_val, timestamp_test, mask = carRome2.load_data(
        T=T, nb_flow=nb_flow, len_closeness=len_c, len_period=len_p, len_trend=len_t, len_test=len_test,
        len_val=len_val, preprocess_name=preprocess_name, meta_data=True, holiday_data=True, meteorol_data=False, datapath=DATAPATH, add_half=True)
    if CACHEDATA:
        cache(fname, X_train_all, Y_train_all, X_train, Y_train, X_val, Y_val, X_test, Y_test,
                external_dim, timestamp_train_all, timestamp_train, timestamp_val, timestamp_test, mask)

# build model
model = build_model('BJ', X_train,  Y_train,  conv_filt=64, kernel_sz=(3,3,3),
                    mask=mask, lstm=500, lstm_number=2, add_external_info=True,
                    lr = 0.0001, save_model_pic=None)

## single-step-prediction no TL
nb_epoch = 100
batch_size = 16
hyperparams_name = '3dclost_roma32x32'
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
model_fname = 'TaxiBJ.c4.p1.t0.iter7.best.noMeteo.h5'
model.load_weights(os.path.join('../best_models', '3DCLoST', model_fname))

# predict
Y_pred = model.predict(X_test)  # compute predictions

# evaluate
score = evaluate(Y_test, Y_pred, mmn)  # evaluate performance

# save to csv
csv_name = os.path.join('results_roma_bergamo', f'TL_taxiBJ_roma32x32_results.csv')
save_to_csv(score, csv_name)

# save real vs predicted
fname = '3dclost_RomaNord32x32.h5'
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
fname = '3dclost_RomaNord32x32_trained.h5'
h5 = h5py.File(os.path.join(path_confronto,fname), 'w')
h5.create_dataset('Y_real', data=Y_test)
h5.create_dataset('Y_pred', data=Y_pred)
h5.create_dataset('timestamps', data=timestamp_test)
h5.create_dataset('max', data=mmn._max)
h5.close()


### 16x8
# params
T = 24  # number of time intervals in one day

len_closeness = len_c = 2  # length of closeness dependent sequence
len_period = len_p = 0  # length of peroid dependent sequence
len_trend = len_t = 1  # length of trend dependent sequence

nb_flow = 2  # there are two types of flows: new-flow and end-flow
# divide data into two subsets: Train & Test,
days_test = 7
len_test = T*days_test
len_val = len_test # no val

map_height, map_width = 16, 8  # grid size

# load data
print("loading data...")
preprocess_name = 'preprocessing_rome16x8_2.pkl'
fname = os.path.join(path_cache, 'Rome16x8_C{}_P{}_T{}_2.h5'.format(
    len_c, len_p, len_t))
if os.path.exists(fname) and CACHEDATA:
    X_train_all, Y_train_all, X_train, Y_train, \
    X_val, Y_val, X_test, Y_test, mmn, external_dim, \
    timestamp_train_all, timestamp_train, timestamp_val, timestamp_test, mask = read_cache(
        fname, preprocess_name)
    print("load %s successfully" % fname)
else:
    X_train_all, Y_train_all, X_train, Y_train, \
    X_val, Y_val, X_test, Y_test, mmn, external_dim, \
    timestamp_train_all, timestamp_train, timestamp_val, timestamp_test, mask = carRome2.load_data(
        T=T, nb_flow=nb_flow, len_closeness=len_c, len_period=len_p, len_trend=len_t, len_test=len_test,
        len_val=len_val, preprocess_name=preprocess_name, meta_data=True, holiday_data=False, datapath=DATAPATH, shape=(16,8))
    if CACHEDATA:
        cache(fname, X_train_all, Y_train_all, X_train, Y_train, X_val, Y_val, X_test, Y_test,
                external_dim, timestamp_train_all, timestamp_train, timestamp_val, timestamp_test, mask)

# build model
model = build_model('NY', X_train,  Y_train, conv_filt=64, kernel_sz=(2,3,3),
                mask=mask, lstm=500, lstm_number=2, add_external_info=True,
                lr=0.0001, save_model_pic=None)

## single-step-prediction no TL
nb_epoch = 100
batch_size = 16
hyperparams_name = '3dclost_roma16x8'
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
model_fname = 'TaxiNYC3.c2.p0.t1.lstm_500.lstmnumber_2.lr_0.00076.batchsize_16.best.h5'
model.load_weights(os.path.join('../best_models', '3DCLoST', model_fname))

# predict
Y_pred = model.predict(X_test)  # compute predictions

# evaluate
score = evaluate(Y_test, Y_pred, mmn)  # evaluate performance

# save to csv
csv_name = os.path.join('results_roma_bergamo', f'TL_taxiNY_roma16x8_results.csv')
save_to_csv(score, csv_name)

# save real vs predicted
fname = '3dclost_RomaNord16x8.h5'
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
fname = '3dclost_RomaNord16x8_trained.h5'
h5 = h5py.File(os.path.join(path_confronto,fname), 'w')
h5.create_dataset('Y_real', data=Y_test)
h5.create_dataset('Y_pred', data=Y_pred)
h5.create_dataset('timestamps', data=timestamp_test)
h5.create_dataset('max', data=mmn._max)
h5.close()
