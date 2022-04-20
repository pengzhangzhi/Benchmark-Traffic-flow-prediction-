from __future__ import print_function
import os
import pickle
import numpy as np
import time
import h5py
import math
from sklearn.model_selection import ParameterGrid
from bayes_opt import BayesianOptimization
import json

import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from deepst.models.STResNet import stresnet
import deepst.metrics as metrics
from deepst.datasets import TaxiNYC
from deepst.evaluation import evaluate

# parameters
DATAPATH = '../data'
nb_epoch = 100  # number of epoch at training stage
# nb_epoch_cont = 150  # number of epoch at training (cont) stage
batch_size = [16, 32, 64]  # batch size
T = 24  # number of time intervals in one day
CACHEDATA = True  # cache data or NOT

lr = [0.00015, 0.00035]  # learning rate
len_closeness = 3  # length of closeness dependent sequence
len_period = 1  # length of peroid dependent sequence
len_trend = 1  # length of trend dependent sequence
len_cpt = [[3,1,1]]
nb_residual_unit = [2,4,6]   # number of residual units

nb_flow = 2  # there are two types of flows: new-flow and end-flow
# divide data into two subsets: Train & Test,
days_test = 7*4
len_test = T*days_test
len_val = 2*len_test

map_height, map_width = 16, 8  # grid size

path_cache = os.path.join(DATAPATH, 'CACHE', 'ST-ResNet')  # cache path
path_result = 'RET'
path_model = 'MODEL'
if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)
if CACHEDATA and os.path.isdir(path_cache) is False:
    os.mkdir(path_cache)

def build_model(len_closeness, len_period, len_trend, nb_flow, map_height, map_width,
                external_dim, nb_residual_unit, bn, bn2=False, save_model_pic=False, lr=0.00015):
    c_conf = (len_closeness, nb_flow, map_height,
              map_width) if len_closeness > 0 else None
    p_conf = (len_period, nb_flow, map_height,
              map_width) if len_period > 0 else None
    t_conf = (len_trend, nb_flow, map_height,
              map_width) if len_trend > 0 else None

    model = stresnet(c_conf=c_conf, p_conf=p_conf, t_conf=t_conf,
                     external_dim=external_dim, nb_residual_unit=nb_residual_unit, bn=bn, bn2=bn2)
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    # model.summary()
    if (save_model_pic):
        from keras.utils.vis_utils import plot_model
        plot_model(model, to_file='TaxiNYC_model.png', show_shapes=True)

    return model

def read_cache(fname):
    mmn = pickle.load(open('preprocessing_taxinyc.pkl', 'rb'))

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

    # load data
print("loading data...")
fname = os.path.join(path_cache, 'TaxiNYC_C{}_P{}_T{}.h5'.format(
    len_closeness, len_period, len_trend))
if os.path.exists(fname) and CACHEDATA:
    X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = read_cache(
        fname)
    print("load %s successfully" % fname)
else:
    X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = TaxiNYC.load_data(
        T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,
        preprocess_name='preprocessing_taxinyc.pkl', meta_data=True,
        meteorol_data=True, holiday_data=True, datapath=DATAPATH)
    if CACHEDATA:
        cache(fname, X_train, Y_train, X_test, Y_test,
              external_dim, timestamp_train, timestamp_test)

print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])
print('=' * 10)


def train_model(lr, batch_size, residual_units, save_results=False, i=''):
    # get discrete parameters
    residual_units = int(residual_units) * 2
    batch_size = 16 * int(batch_size)
    # kernel_size = int(kernel_size)
    lr = round(lr,5)

    # build model
    tf.keras.backend.set_image_data_format('channels_first')
    model = build_model(len_closeness, len_period, len_trend, nb_flow, map_height,
                        map_width, external_dim, residual_units,
                        bn=True,
                        bn2=True,
                        save_model_pic=False,
                        lr=lr
                        )
    # model.summary()
    hyperparams_name = 'TaxiNYC{}.c{}.p{}.t{}.resunits_{}.lr_{}.batchsize_{}'.format(
        i, len_closeness, len_period, len_trend, residual_units,
        lr, batch_size)
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
    history = model.fit(X_train, Y_train,
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
        print('evaluating using the model that has the best loss on the valid set')
        model.load_weights(fname_param)  # load best weights for current iteration

        Y_pred = model.predict(X_test)  # compute predictions

        score = evaluate(Y_test, Y_pred, mmn, rmse_factor=1)  # evaluate performance

        # save to csv
        csv_name = os.path.join('results', 'star_taxiNYC_results.csv')
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

# bayes opt is a maximization algorithm, to minimize validation_loss, return 1-this
    bayes_opt_score = 1.0 - score[1]

    return bayes_opt_score

# bayesian optimization
optimizer = BayesianOptimization(f=train_model,
                                 pbounds={'residual_units': (1, 3.999), # *2
                                          'lr': (0.0001,0.001),
                                          'batch_size': (1, 2.999), # *16
                                        #   'kernel_size': (3, 5.999)
                                 },
                                 verbose=2)

optimizer.maximize(init_points=2, n_iter=10)

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
params_fname = 'star_taxiNYC_best_params.json'
with open(os.path.join('results', params_fname), 'w') as f:
    json.dump(params, f, indent=2)
# with open(os.path.join('results', params_fname), 'r') as f:
#     params = json.load(f)
for i in range(0, 1):
    train_model(residual_units=params['residual_units'],
                lr=params['lr'],
                batch_size=params['batch_size'],
                # kernel_size=params['kernel_size'],
                save_results=True,
                i=i)
