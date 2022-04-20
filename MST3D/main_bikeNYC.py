from __future__ import print_function
import os
import sys
import pickle
import time
import numpy as np
import h5py
import math

import tensorflow as tf
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

import deepst.metrics as metrics
from deepst.datasets import BikeNYC
from deepst.model import mst3d_nyc
from deepst.evaluation import evaluate

# np.random.seed(1337)  # for reproducibility

# parameters
DATAPATH = '../data'
CACHEDATA = True  # cache data or NOT
path_cache = os.path.join(DATAPATH, 'CACHE', 'MST3D')  # cache path
nb_epoch = 200  # number of epoch at training stage
nb_epoch_cont = 100  # number of epoch at training (cont) stage
batch_size = 64  # batch size
T = 24  # number of time intervals in one day
lr = 0.0002  # learning rate
len_closeness = 4  # length of closeness dependent sequence - should be 6
len_period = 4  # length of peroid dependent sequence
len_trend = 4  # length of trend dependent sequence

nb_flow = 2  # there are two types of flows: inflow and outflow

# divide data into two subsets: Train & Test, of which the test set is the
# last 10 days
days_test = 10
len_test = T * days_test
map_height, map_width = 16, 8  # grid size
# For NYC Bike data, there are 81 available grid-based areas, each of
# which includes at least ONE bike station. Therefore, we modify the final
# RMSE by multiplying the following factor (i.e., factor).
nb_area = 81
m_factor = math.sqrt(1. * map_height * map_width / nb_area)

path_result = 'RET'
path_model = 'MODEL'
if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)
if CACHEDATA and os.path.isdir(path_cache) is False:
    os.mkdir(path_cache)

def build_model(save_model_pic=False):
    model = mst3d_nyc(len_closeness, len_period, len_trend, nb_flow, map_height, map_width, external_dim)
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    # model.summary()
    if (save_model_pic):
        from keras.utils.vis_utils import plot_model
        plot_model(model, to_file='BikeNYC_model.png', show_shapes=True)
    return model

def read_cache(fname):
    mmn = pickle.load(open('preprocessing_bikenyc.pkl', 'rb'))

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

# main code
# load data
print("loading data...")
ts = time.time()
fname = os.path.join(path_cache, 'BikeNYC_C{}_P{}_T{}.h5'.format(
    len_closeness, len_period, len_trend))
if os.path.exists(fname) and CACHEDATA:
    X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = read_cache(
        fname)
    print("load %s successfully" % fname)
else:
    X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = BikeNYC.load_data(
        T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,
        preprocess_name='preprocessing_bikenyc.pkl', meta_data=True, datapath=DATAPATH)
    if CACHEDATA:
        cache(fname, X_train, Y_train, X_test, Y_test,
              external_dim, timestamp_train, timestamp_test)

print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])
print("\nelapsed time (loading data): %.3f seconds\n" % (time.time() - ts))

print('=' * 10)

# training-test-evaluation iterations
for i in range(0,10):
    print('=' * 10)
    print("compiling model...")

    # lr_callback = LearningRateScheduler(lrschedule)

    # build model
    model = build_model(save_model_pic=False)

    hyperparams_name = 'BikeNYC.c{}.p{}.t{}.iter{}'.format(
        len_closeness, len_period, len_trend, i)
    fname_param = os.path.join(path_model, '{}.best.h5'.format(hyperparams_name))
    print(hyperparams_name)

    early_stopping = EarlyStopping(monitor='val_rmse', patience=25, mode='min')
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')

    print('=' * 10)
    # train model
    np.random.seed(i*18)
    tf.random.set_seed(i*18)
    print("training model...")
    history = model.fit(X_train, Y_train,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        validation_split=0.1,
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=0)
    model.save_weights(os.path.join(
        path_model, '{}.h5'.format(hyperparams_name)), overwrite=True)
    pickle.dump((history.history), open(os.path.join(
        path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))

    print('=' * 10)

    # evaluate model
    print('evaluating using the model that has the best loss on the valid set')
    model.load_weights(fname_param) # load best weights for current iteration
    
    Y_pred = model.predict(X_test) # compute predictions

    score = evaluate(Y_test, Y_pred, mmn, rmse_factor=1) # evaluate performance

    # save to csv
    csv_name = os.path.join('results','mst3d_bikeNYC_results.csv')
    if not os.path.isfile(csv_name):
        if os.path.isdir('results') is False:
            os.mkdir('results')
        with open(csv_name, 'a', encoding = "utf-8") as file:
            file.write('iteration,'
                       'rsme_in,rsme_out,rsme_tot,'
                       'mape_in,mape_out,mape_tot,'
                       'ape_in,ape_out,ape_tot'
                       )
            file.write("\n")
            file.close()
    with open(csv_name, 'a', encoding = "utf-8") as file:
        file.write(f'{i},{score[0]},{score[1]},{score[2]},{score[3]},'
                   f'{score[4]},{score[5]},{score[6]},{score[7]},{score[8]}'
                  )
        file.write("\n")
        file.close()
    K.clear_session()