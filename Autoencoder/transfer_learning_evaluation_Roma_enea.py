import time
import os
import h5py
from keras.callbacks import ModelCheckpoint

from utils import cache, read_cache
from src.carRome import load_data
from src.streednet import build_model
from src.evaluation import evaluate
import tensorflow as tf


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
DATAPATH = '../data'
CACHEDATA = True
T = 24*2  # number of time intervals in one day

len_closeness = 4  # length of closeness dependent sequence
len_period = 0  # length of peroid dependent sequence
len_trend = 0  # length of trend dependent sequence

nb_flow = 2  # there are two types of flows: in-flow and out-flow
days_test = 7
len_test = T * days_test
len_val = len_test # no validation

map_height, map_width = 32, 32  # grid size


# load data
cache_folder = 'Autoencoder/model3'
path_cache = os.path.join(DATAPATH, 'CACHE', cache_folder)  # cache path
if CACHEDATA and os.path.isdir(path_cache) is False:
    os.mkdir(path_cache)
if os.path.isdir('results') is False:
    os.mkdir('results')


# load data
print("loading data...")
preprocess_name = 'preprocess_rome.pkl'
fname = os.path.join(path_cache, 'Rome_C{}_P{}_T{}.h5'.format(
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
    timestamp_train_all, timestamp_train, timestamp_val, timestamp_test = load_data(
        T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,
        len_val=len_val, preprocess_name=preprocess_name, meta_data=True, datapath=DATAPATH)
    if CACHEDATA:
        cache(fname, X_train_all, Y_train_all, X_train, Y_train, X_val, Y_val, X_test, Y_test,
              external_dim, timestamp_train_all, timestamp_train, timestamp_val, timestamp_test)

print(external_dim)
print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])

# build model
model = build_model(
    len_closeness, len_period, len_trend, nb_flow, map_height, map_width,
    external_dim=external_dim,
    encoder_blocks=3,
    filters=[64,64,64,64,16],
    kernel_size=3,
    num_res=2
)

## single-step-prediction no TL
nb_epoch = 250
batch_size = 16
hyperparams_name = 'model3_roma32x32'
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
score = evaluate(Y_test, Y_pred, mmn)  # evaluate performance

# save to csv
csv_name = os.path.join('results', f'roma32x32_results.csv')
save_to_csv(score, csv_name)


## TL without re-training
# load weights
model_fname = 'model3resunit_doppia_attention.TaxiBJ1.c4.p0.t0.encoderblocks_3.kernel_size_3.lr_0.0007.batchsize_16.noMeteo.best.h5'
model.load_weights(os.path.join('../best_models', 'model3', model_fname))

# predict
Y_pred = model.predict(X_test)  # compute predictions

# evaluate
score = evaluate(Y_test, Y_pred, mmn)  # evaluate performance

# save to csv
csv_name = os.path.join('results', f'TL_taxiBJ_roma32x32_results.csv')
save_to_csv(score, csv_name)

# save real vs predicted
fname = 'model3_RomaNord32x32.h5'
h5 = h5py.File(fname, 'w')
h5.create_dataset('Y_real', data=Y_test)
h5.create_dataset('Y_pred', data=Y_pred)
h5.create_dataset('timestamps', data=timestamp_test)
h5.create_dataset('max', data=mmn._max)
h5.close()

## TL with re-training
# for l in model.layers[:-28]:
#     l.trainable = False
nb_epoch = 250
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
score = evaluate(Y_test, Y_pred, mmn)  # evaluate performance

# save to csv
csv_name = os.path.join('results', f'TL_taxiBJ_roma32x32_training_results.csv')
save_to_csv(score, csv_name)

# save real vs predicted
fname = 'model3_RomaNord32x32_trained.h5'
h5 = h5py.File(fname, 'w')
h5.create_dataset('Y_real', data=Y_test)
h5.create_dataset('Y_pred', data=Y_pred)
h5.create_dataset('timestamps', data=timestamp_test)
h5.create_dataset('max', data=mmn._max)
h5.close()


### 16x8
T = 24
len_closeness = 4  # length of closeness dependent sequence
len_period = 0  # length of peroid dependent sequence
len_trend = 0  # length of trend dependent sequence
map_height, map_width = 16, 8  # grid size
days_test = 7
len_test = T * days_test
len_val = len_test # no validation

# load data
cache_folder = 'Autoencoder/model3'
path_cache = os.path.join(DATAPATH, 'CACHE', cache_folder)  # cache path
if CACHEDATA and os.path.isdir(path_cache) is False:
    os.mkdir(path_cache)

# load data
print("loading data...")
preprocess_name = 'preprocess_rome16x8.pkl'
fname = os.path.join(path_cache, 'Rome16x8_C{}_P{}_T{}.h5'.format(
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
    timestamp_train_all, timestamp_train, timestamp_val, timestamp_test = load_data(
        T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,
        len_val=len_val, preprocess_name=preprocess_name, meta_data=True, meteorol_data=True, holiday_data=True, datapath=DATAPATH, shape=(16,8))
    if CACHEDATA:
        cache(fname, X_train_all, Y_train_all, X_train, Y_train, X_val, Y_val, X_test, Y_test,
              external_dim, timestamp_train_all, timestamp_train, timestamp_val, timestamp_test)

print(external_dim)
print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])

# build model
model = build_model(
    len_closeness, len_period, len_trend, nb_flow, map_height, map_width,
    external_dim=external_dim,
    encoder_blocks=2,
    filters=[64,64,64,16],
    kernel_size=3,
    num_res=2
)

## single-step-prediction no TL
nb_epoch = 250
batch_size = 16
hyperparams_name = 'model3_roma16x8'
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
score = evaluate(Y_test, Y_pred, mmn)  # evaluate performance

# save to csv
csv_name = os.path.join('results', f'roma16x8_results.csv')
save_to_csv(score, csv_name)


## TL without re-training
# load weights
model_fname = 'model3resunit_doppia_attention.TaxiNYC5.c4.p0.t0.encoderblocks_2.kernel_size_3.lr_0.00086.batchsize_48.best.h5'
model.load_weights(os.path.join('../best_models', 'model3', model_fname))

# predict
Y_pred = model.predict(X_test)  # compute predictions

# evaluate
score = evaluate(Y_test, Y_pred, mmn)  # evaluate performance

# save to csv
csv_name = os.path.join('results', f'TL_taxiNY_roma16x8_results.csv')
save_to_csv(score, csv_name)

# save real vs predicted
fname = 'model3_RomaNord16x8.h5'
h5 = h5py.File(fname, 'w')
h5.create_dataset('Y_real', data=Y_test)
h5.create_dataset('Y_pred', data=Y_pred)
h5.create_dataset('timestamps', data=timestamp_test)
h5.create_dataset('max', data=mmn._max)
h5.close()

## TL with re-training
nb_epoch = 250
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
score = evaluate(Y_test, Y_pred, mmn)  # evaluate performance

# save to csv
csv_name = os.path.join('results', f'TL_taxiNY_roma16x8_training_results.csv')
save_to_csv(score, csv_name)

# save real vs predicted
fname = 'model3_RomaNord16x8_trained.h5'
h5 = h5py.File(fname, 'w')
h5.create_dataset('Y_real', data=Y_test)
h5.create_dataset('Y_pred', data=Y_pred)
h5.create_dataset('timestamps', data=timestamp_test)
h5.create_dataset('max', data=mmn._max)
h5.close()
