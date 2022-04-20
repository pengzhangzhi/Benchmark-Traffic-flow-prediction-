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
from src.datasets import BikeNYC
from src.evaluation import evaluate
from cache_utils import cache, read_cache

# note: this code could not work without a GPU

# np.random.seed(1337)  # for reproducibility

# parameters
DATAPATH = '../data' 
nb_epoch = 150  # number of epoch at training stage
# nb_epoch_cont = 150  # number of epoch at training (cont) stage
batch_size = 64  # batch size
T = 24  # number of time intervals in one day
CACHEDATA = True  # cache data or NOT

lr = 0.001  # learning rate
len_c = 2  # length of closeness dependent sequence
len_p = 0  # length of peroid dependent sequence
len_t = 1  # length of trend dependent sequence

nb_flow = 2  # there are two types of flows: new-flow and end-flow
# divide data into two subsets: Train & Test, of which the test set is the
# last 10 days
days_test = 10
len_test = T*days_test
len_val = 2*len_test

map_height, map_width = 16, 8  # grid size
# For NYC Bike data, there are 81 available grid-based areas, each of
# which includes at least ONE bike station. Therefore, we modify the final
# RMSE by multiplying the following factor (i.e., factor).
nb_area = 81
m_factor = math.sqrt(1. * map_height * map_width / nb_area)
# print('factor: ', m_factor)

path_cache = os.path.join(DATAPATH, 'CACHE', '3D-CLoST')  # cache path
path_result = 'RET'
path_model = 'MODEL'
if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)
if CACHEDATA and os.path.isdir(path_cache) is False:
    os.mkdir(path_cache)

# load data
print("loading data...")
preprocess_name = 'preprocessing_bikenyc.pkl'
fname = os.path.join(path_cache, 'BikeNYC_C{}_P{}_T{}.h5'.format(
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
    timestamp_train_all, timestamp_train, timestamp_val, timestamp_test, mask = BikeNYC.load_data(
        T=T, nb_flow=nb_flow, len_closeness=len_c, len_period=len_p, len_trend=len_t, len_test=len_test,
        len_val=len_val, preprocess_name=preprocess_name, meta_data=True, datapath=DATAPATH)
    if CACHEDATA:
        cache(fname, X_train_all, Y_train_all, X_train, Y_train, X_val, Y_val, X_test, Y_test,
                external_dim, timestamp_train_all, timestamp_train, timestamp_val, timestamp_test, mask)

# print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])

# training-test-evaluation iterations
for i in range(0,10):
    print('=' * 10)
    print("compiling model...")

    # build model
    model = build_model('NY', X_train,  Y_train, conv_filt=64, kernel_sz=(2,3,3), 
                    mask=mask, lstm=350, lstm_number=2, add_external_info=True,
                    lr=0.001, save_model_pic=None)

    hyperparams_name = 'BikeNYC.c{}.p{}.t{}.iter{}'.format(
        len_c, len_p, len_t, i)
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
                        validation_data=(X_val,Y_val),
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
    csv_name = os.path.join('results','3DCLoST_bikeNYC_results.csv')
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
