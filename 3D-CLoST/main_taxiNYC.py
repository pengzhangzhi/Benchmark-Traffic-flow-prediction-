from __future__ import print_function
import os
import _pickle as pickle
import numpy as np
import math
import h5py
import time
import json
from bayes_opt import BayesianOptimization

import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from src.model import build_model
import src.metrics as metrics
from src.datasets import TaxiNYC
from src.evaluation import evaluate
from cache_utils import cache, read_cache


# tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:  # Currently, memory growth needs to be the same across GPUs
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)  # Memory growth must be set before GPUs have been initialized

# parameters
DATAPATH = '../data' 
nb_epoch = 100  # number of epoch at training stage
# nb_epoch_cont = 150  # number of epoch at training (cont) stage
T = 24  # number of time intervals in one day
CACHEDATA = True  # cache data or NOT

len_closeness = len_c = 2  # length of closeness dependent sequence
len_period = len_p = 0  # length of peroid dependent sequence
len_trend = len_t = 1  # length of trend dependent sequence
# len_cpt = [[2,0,1]]
# batch_size = [16, 64]  # batch size
# lr = [0.0015, 0.00015]  # learning rate
# lstm = [350, 500]
# lstm_number = [2, 3]

nb_flow = 2  # there are two types of flows: new-flow and end-flow
# divide data into two subsets: Train & Test, 
days_test = 7*4
len_test = T*days_test
len_val = 2*len_test

map_height, map_width = 16, 8  # grid size

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
fname = os.path.join(path_cache, 'TaxiNYC_C{}_P{}_T{}.h5'.format(
    len_c, len_p, len_t))
if os.path.exists(fname) and CACHEDATA:
    X_train_all, Y_train_all, X_train, Y_train, \
    X_val, Y_val, X_test, Y_test, mmn, external_dim, \
    timestamp_train_all, timestamp_train, timestamp_val, timestamp_test, mask = read_cache(
        fname, 'preprocessing_taxinyc.pkl')
    print("load %s successfully" % fname)
else:
    X_train_all, Y_train_all, X_train, Y_train, \
    X_val, Y_val, X_test, Y_test, mmn, external_dim, \
    timestamp_train_all, timestamp_train, timestamp_val, timestamp_test, mask = TaxiNYC.load_data(
        T=T, nb_flow=nb_flow, len_closeness=len_c, len_period=len_p, len_trend=len_t, len_test=len_test,
        len_val=len_val, preprocess_name='preprocessing_taxinyc.pkl', meta_data=True, datapath=DATAPATH)
    if CACHEDATA:
        cache(fname, X_train_all, Y_train_all, X_train, Y_train, X_val, Y_val, X_test, Y_test,
                external_dim, timestamp_train_all, timestamp_train, timestamp_val, timestamp_test, mask)

# print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])
print('=' * 10)

def train_model(lr, batch_size, lstm, lstm_number, save_results=False, i=''):
    # get discrete parameters
    lstm = 350 if lstm < 1 else 500
    lstm_number = int(lstm_number)
    batch_size = 16 if batch_size < 2 else 64
    lr = round(lr,5)

    # build model
    model = build_model('NY', X_train,  Y_train, conv_filt=64, kernel_sz=(2,3,3), 
                    mask=mask, lstm=lstm, lstm_number=lstm_number, add_external_info=True,
                    lr=lr, save_model_pic=None)

    # model.summary()
    hyperparams_name = 'TaxiNYC{}.c{}.p{}.t{}.lstm_{}.lstmnumber_{}.lr_{}.batchsize_{}'.format(
            i, len_c, len_p, len_t, lstm, lstm_number, lr, batch_size)
    fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))

    early_stopping = EarlyStopping(monitor='val_rmse', patience=25, mode='min')
    # lr_callback = LearningRateScheduler(lrschedule)
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')

    # train model
    print("training model...")
    ts = time.time()
    if (i):
        print(f'Iteration {i}')
        np.random.seed(i * 18)
        tf.random.set_seed(i * 18)
    history = model.fit(X_train_all, Y_train_all,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        validation_data=(X_test, Y_test),
                        # callbacks=[early_stopping, model_checkpoint],
                        # callbacks=[model_checkpoint, lr_callback],
                        callbacks=[model_checkpoint],
                        verbose=2)
    model.save_weights(os.path.join(
        'MODEL', '{}.h5'.format(hyperparams_name)), overwrite=True)
    pickle.dump((history.history), open(os.path.join(
        path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))
    print("\nelapsed time (training): %.3f seconds\n" % (time.time() - ts))

    # evaluate
    model.load_weights(fname_param)
    score = model.evaluate(
        X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))

    if (save_results):
        score = test_model(i, model, fname_param)

    # bayes opt is a maximization algorithm, to minimize validation_loss, return 1-this
    bayes_opt_score = 1.0 - score[1]

    return bayes_opt_score

def test_model(i, model, fname_param):
    print('evaluating using the model that has the best loss on the valid set')
    model.load_weights(fname_param)  # load best weights for current iteration

    Y_pred = model.predict(X_test)  # compute predictions

    score = evaluate(Y_test, Y_pred, mmn, rmse_factor=1)  # evaluate performance

        # save to csv
    csv_name = os.path.join('results', '3dclost_taxiNYC_results.csv')
    if not os.path.isfile(csv_name):
        if os.path.isdir('results') is False:
            os.mkdir('results')
        with open(csv_name, 'a', encoding="utf-8") as file:
            file.write('iteration,'
                           'rsme_in,rsme_out,rsme_tot,'
                           'mape_in,mape_out,mape_tot,'
                           'ape_in,ape_out,ape_tot,mae_tot'
                           )
            file.write("\n")
            file.close()
    with open(csv_name, 'a', encoding="utf-8") as file:
        file.write(f'{i},{score[0]},{score[1]},{score[2]},{score[3]},'
                       f'{score[4]},{score[5]},{score[6]},{score[7]},{score[8]},{score[9]}'
                       )
        file.write("\n")
        file.close()
    K.clear_session()
    return score

# bayesian optimization
optimizer = BayesianOptimization(f=train_model,
                                 pbounds={'lstm': (0, 1.999), # 350 if smaller than 1 else 500
                                          'lstm_number': (2, 3.999),
                                          'lr': (0.0001,0.001),
                                          'batch_size': (1, 2.999), # 16 if smaller than 2 else 64
                                        #   'kernel_size': (3, 5.999)
                                 },
                                 verbose=2)

optimizer.maximize(init_points=2, n_iter=12)

# training-test-evaluation iterations with best params
if os.path.isdir('results') is False:
    os.mkdir('results')
targets = [e['target'] for e in optimizer.res]
bs_fname = 'bs_taxiNYC.json'
with open(os.path.join('results', bs_fname), 'w') as f:
    json.dump(optimizer.res, f, indent=2)
best_index = targets.index(max(targets))
params = optimizer.res[best_index]['params']
# save best params
params_fname = '3dclost_taxiNYC_best_params.json'
with open(os.path.join('results', params_fname), 'w') as f:
    json.dump(params, f, indent=2)
# with open(os.path.join('results', params_fname), 'r') as f:
#     params = json.load(f)
for i in range(0, 1):
    train_model(lstm=params['lstm'],
                lstm_number=params['lstm_number'],
                lr=params['lr'],
                batch_size=params['batch_size'],
                save_results=True,
                i=i)
