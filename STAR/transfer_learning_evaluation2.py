from __future__ import print_function
import sys
import pickle as pickle
import time
import h5py
import os

from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from star.model import *
import star.metrics as metrics
from star import carRome2
from star.evaluation import evaluate


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

def build_model(external_dim, save_model_pic=False, bn=False):
    c_conf = (len_closeness, nb_flow, map_height,
              map_width) if len_closeness > 0 else None
    p_conf = (len_period, nb_flow, map_height,
              map_width) if len_period > 0 else None
    t_conf = (len_trend, nb_flow, map_height,
              map_width) if len_trend > 0 else None
    model = STAR(c_conf=c_conf, p_conf=p_conf, t_conf=t_conf,
                     external_dim=external_dim, nb_residual_unit=nb_residual_unit, bn=bn, bn2=bn)
    # sgd = SGD(lr=lr, momentum=0.9, decay=5e-4, nesterov=True)
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    # model.summary()
    if (save_model_pic):
        from keras.utils.vis_utils import plot_model
        plot_model(model, to_file='TaxiBJ_model.png', show_shapes=True)

    return model

def read_cache(fname, preprocess_name):
    mmn = pickle.load(open(preprocess_name, 'rb'))

    f = h5py.File(fname, 'r')
    num = int(f['num'].value)
    X_train_all, Y_train_all, X_train, Y_train, X_val, Y_val, X_test, Y_test = [], [], [], [], [], [], [], []
    for i in range(num):
        X_train_all.append(f['X_train_all_%i' % i].value)
        X_train.append(f['X_train_%i' % i].value)
        X_val.append(f['X_val_%i' % i].value)
        X_test.append(f['X_test_%i' % i].value)
    Y_train_all = f['Y_train_all'].value
    Y_train = f['Y_train'].value
    Y_val = f['Y_val'].value
    Y_test = f['Y_test'].value
    external_dim = f['external_dim'].value
    timestamp_train_all = f['T_train_all'].value
    timestamp_train = f['T_train'].value
    timestamp_val = f['T_val'].value
    timestamp_test = f['T_test'].value
    f.close()

    return X_train_all, Y_train_all, X_train, Y_train, X_val, Y_val, X_test, Y_test, mmn, external_dim, timestamp_train_all, timestamp_train, timestamp_val, timestamp_test

def cache(fname, X_train_all, Y_train_all, X_train, Y_train, X_val, Y_val, X_test, Y_test, external_dim, timestamp_train_all, timestamp_train, timestamp_val, timestamp_test):
    h5 = h5py.File(fname, 'w')
    h5.create_dataset('num', data=len(X_train_all))

    for i, data in enumerate(X_train_all):
        h5.create_dataset('X_train_all_%i' % i, data=data)
    for i, data in enumerate(X_train):
        h5.create_dataset('X_train_%i' % i, data=data)
    for i, data in enumerate(X_val):
        h5.create_dataset('X_val_%i' % i, data=data)
    # for i, data in enumerate(Y_train):
    for i, data in enumerate(X_test):
        h5.create_dataset('X_test_%i' % i, data=data)

    h5.create_dataset('Y_train_all', data=Y_train_all)
    h5.create_dataset('Y_train', data=Y_train)
    h5.create_dataset('Y_val', data=Y_val)
    h5.create_dataset('Y_test', data=Y_test)
    external_dim = -1 if external_dim is None else int(external_dim)
    h5.create_dataset('external_dim', data=external_dim)
    h5.create_dataset('T_train_all', data=timestamp_train_all)
    h5.create_dataset('T_train', data=timestamp_train)
    h5.create_dataset('T_val', data=timestamp_val)
    h5.create_dataset('T_test', data=timestamp_test)
    h5.close()

### 32x32
# params
DATAPATH = '../data'
CACHEDATA = True  # cache data or NOT
T = 24*2  # number of time intervals in one day
lr = 0.0001 # learning rate

len_closeness = 3 # length of closeness dependent sequence
len_period = 1 # length of peroid dependent sequence
len_trend = 1 # length of trend dependent sequence
nb_residual_unit = 6
nb_flow = 2
days_test = 7
len_test = T*days_test
len_val = len_test # no val
map_height, map_width = 32, 32  # grid size

# path_log = 'log_BJ'
# muilt_step = False

path_cache = os.path.join(DATAPATH, 'CACHE', 'STAR')  # cache path
if CACHEDATA and os.path.isdir(path_cache) is False:
    os.mkdir(path_cache)

# load data
print("loading data...")
ts = time.time()
preprocess_name = 'preprocessing_rome_2.pkl'
fname = os.path.join(path_cache, 'Rome_C{}_P{}_T{}_2.h5'.format(
    len_closeness, len_period, len_trend))
if os.path.exists(fname) and CACHEDATA:
    X_train_all, Y_train_all, X_train, Y_train, \
    X_val, Y_val, X_test, Y_test, mmn, external_dim, \
    timestamp_train_all, timestamp_train, timestamp_val, timestamp_test = read_cache(
        fname, preprocess_name)
    print("load %s successfully" % fname)
else:
    X_train_all, Y_train_all, X_train, Y_train, \
    X_val, Y_val, X_test, Y_test, mmn, external_dim, \
    timestamp_train_all, timestamp_train, timestamp_val, timestamp_test = carRome2.load_data(
        T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,
        len_val=len_val, preprocess_name=preprocess_name, meta_data=True, meteorol_data=False, holiday_data=True, datapath=DATAPATH)
    if CACHEDATA:
        cache(fname, X_train_all, Y_train_all, X_train, Y_train, X_val, Y_val, X_test, Y_test,
              external_dim, timestamp_train_all, timestamp_train, timestamp_val, timestamp_test)
# print(external_dim)
print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])
print("\nelapsed time (loading data): %.3f seconds\n" % (time.time() - ts))
print('=' * 10)

# build model
tf.keras.backend.set_image_data_format('channels_first') # set the image format
tf.keras.backend.image_data_format() # returns the default image data format convention
model = build_model(external_dim, save_model_pic=False)

## single-step-prediction no TL
nb_epoch = 100
batch_size = 16
hyperparams_name = 'star_roma32x32'
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
model_fname = 'TaxiBJ.c3.p1.t1.resunit6.lr0.00015.iter8.noMeteo.cont.best.h5'
model.load_weights(os.path.join('../best_models', 'STAR', model_fname))

# predict
Y_pred = model.predict(X_test)  # compute predictions
# evaluate
score = evaluate(Y_test, Y_pred, mmn)  # evaluate performance

# save to csv
csv_name = os.path.join('results_roma_bergamo', f'TL_taxiBJ_roma32x32_results.csv')
save_to_csv(score, csv_name)

# save real vs predicted
fname = 'star_RomaNord32x32.h5'
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
fname = 'star_RomaNord32x32_trained.h5'
h5 = h5py.File(os.path.join(path_confronto,fname), 'w')
h5.create_dataset('Y_real', data=Y_test)
h5.create_dataset('Y_pred', data=Y_pred)
h5.create_dataset('timestamps', data=timestamp_test)
h5.create_dataset('max', data=mmn._max)
h5.close()


### 16x8
DATAPATH = '../data'
CACHEDATA = True  # cache data or NOT
T = 24  # number of time intervals in one day
lr = 0.0001 # learning rate

len_closeness = 3 # length of closeness dependent sequence
len_period = 1 # length of peroid dependent sequence
len_trend = 1 # length of trend dependent sequence
nb_residual_unit = 4
nb_flow = 2
days_test = 7
len_test = T*days_test
len_val = len_test # no val
map_height, map_width = 16, 8  # grid size

# load data
print("loading data...")
ts = time.time()
preprocess_name = 'preprocessing_rome16x8_2.pkl'
fname = os.path.join(path_cache, 'Rome16x8_C{}_P{}_T{}_2.h5'.format(
    len_closeness, len_period, len_trend))
if os.path.exists(fname) and CACHEDATA:
    X_train_all, Y_train_all, X_train, Y_train, \
    X_val, Y_val, X_test, Y_test, mmn, external_dim, \
    timestamp_train_all, timestamp_train, timestamp_val, timestamp_test = read_cache(
        fname, preprocess_name)
    print("load %s successfully" % fname)
else:
    X_train_all, Y_train_all, X_train, Y_train, \
    X_val, Y_val, X_test, Y_test, mmn, external_dim, \
    timestamp_train_all, timestamp_train, timestamp_val, timestamp_test = carRome2.load_data(
        T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,
        len_val=len_val, preprocess_name=preprocess_name, meta_data=True, meteorol_data=True, holiday_data=True, datapath=DATAPATH, shape=(16,8))
    if CACHEDATA:
        cache(fname, X_train_all, Y_train_all, X_train, Y_train, X_val, Y_val, X_test, Y_test,
              external_dim, timestamp_train_all, timestamp_train, timestamp_val, timestamp_test)
# print(external_dim)
print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])
print("\nelapsed time (loading data): %.3f seconds\n" % (time.time() - ts))
print('=' * 10)

# build model
tf.keras.backend.set_image_data_format('channels_first') # set the image format
tf.keras.backend.image_data_format() # returns the default image data format convention
model = build_model(external_dim, save_model_pic=False, bn=True)

## single-step-prediction no TL
nb_epoch = 100
batch_size = 16
hyperparams_name = 'star_roma16x8'
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
model_fname = 'TaxiNYC4.c3.p1.t1.resunits_4.lr_0.0001.batchsize_16.best.h5'
model.load_weights(os.path.join('../best_models', 'STAR', model_fname))

# predict
Y_pred = model.predict(X_test)  # compute predictions
# evaluate
score = evaluate(Y_test, Y_pred, mmn)  # evaluate performance

# save to csv
csv_name = os.path.join('results_roma_bergamo', f'TL_taxiNY_roma16x8_results.csv')
save_to_csv(score, csv_name)

# save real vs predicted
fname = 'star_RomaNord16x8.h5'
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
fname = 'star_RomaNord16x8_trained.h5'
h5 = h5py.File(os.path.join(path_confronto,fname), 'w')
h5.create_dataset('Y_real', data=Y_test)
h5.create_dataset('Y_pred', data=Y_pred)
h5.create_dataset('timestamps', data=timestamp_test)
h5.create_dataset('max', data=mmn._max)
h5.close()
