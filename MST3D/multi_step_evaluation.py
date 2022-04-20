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

import deepst.metrics as metrics
from deepst.datasets import TaxiBJ, TaxiNYC, BikeNYC
from deepst.model import mst3d_bj, mst3d_nyc
from deepst.evaluation import evaluate
from deepst.multi_step import multi_step_2D


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


def build_model(city, len_closeness, len_period, len_trend, nb_flow, map_height, map_width, external_dim, lr=0.0001, save_model_pic=False):
    create_model = mst3d_bj if city=='BJ' else mst3d_nyc
    model = create_model(len_closeness, len_period, len_trend, nb_flow, map_height, map_width, external_dim)
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    # model.summary()
    if (save_model_pic):
        from keras.utils.vis_utils import plot_model
        plot_model(model, to_file='TaxiBJ_model.png', show_shapes=True)
    return model

def read_cache(fname, preprocess_fname):
    mmn = pickle.load(open(preprocess_fname, 'rb'))

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

def taxibj_evaluation():
    # parameters
    DATAPATH = '../data'
    CACHEDATA = True  # cache data or NOT
    T = 48  # number of time intervals in one day
    lr = 0.0002  # learning rate
    len_closeness = 4  # length of closeness dependent sequence - should be 6
    len_period = 4  # length of peroid dependent sequence
    len_trend = 4  # length of trend dependent sequence

    nb_flow = 2
    days_test = 7 * 4
    len_test = T * days_test
    map_height, map_width = 32, 32  # grid size

    path_cache = os.path.join(DATAPATH, 'CACHE', 'MST3D')  # cache path
    if CACHEDATA and os.path.isdir(path_cache) is False:
        os.mkdir(path_cache)

    # load data
    print("loading data...")
    ts = time.time()
    fname = os.path.join(path_cache, 'TaxiBJ_C{}_P{}_T{}.h5'.format(
        len_closeness, len_period, len_trend))
    if os.path.exists(fname) and CACHEDATA:
        X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = read_cache(
            fname, 'preprocessing.pkl')
        print("load %s successfully" % fname)
    else:
        X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = TaxiBJ.load_data(
            T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,
            preprocess_name='preprocessing.pkl', meta_data=True, meteorol_data=True, holiday_data=True, datapath=DATAPATH)
        if CACHEDATA:
            cache(fname, X_train, Y_train, X_test, Y_test,
                  external_dim, timestamp_train, timestamp_test)

    print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])
    print("\nelapsed time (loading data): %.3f seconds\n" % (time.time() - ts))

    print('=' * 10)
    mmn._max = 1292 # just to be sure it's correct

    # build model
    model = build_model('BJ', len_closeness, len_period, len_trend, nb_flow,
                        map_height, map_width, external_dim)
    model_fname = 'TaxiBJ.c4.p4.t4.iter6.best.h5'
    model.load_weights(os.path.join('../best_models', 'MST3D', model_fname))

    # evaluate and save results
    dict_multi_score = multi_step_2D(model, X_test, Y_test, mmn, len_closeness, step=5)

    for i in range(len(dict_multi_score)):
        csv_name = os.path.join('results', f'taxibj_step{i+1}.csv')
        save_to_csv(dict_multi_score[i], csv_name)


def taxiny_evaluation():
    # parameters
    DATAPATH = '../data'
    CACHEDATA = True  # cache data or NOT
    T = 24  # number of time intervals in one day
    lr = 0.00035 # learning rate
    len_closeness = len_c = 4  # length of closeness dependent sequence - should be 6
    len_period = len_p = 4  # length of peroid dependent sequence
    len_trend = len_t = 4  # length of trend dependent sequence
    nb_flow = 2  # there are two types of flows: inflow and outflow

    days_test = 7*4
    len_test = T * days_test
    map_height, map_width = 16, 8  # grid size

    path_cache = os.path.join(DATAPATH, 'CACHE', 'MST3D')  # cache path
    if CACHEDATA and os.path.isdir(path_cache) is False:
        os.mkdir(path_cache)

    # load data
    print("loading data...")
    fname = os.path.join(path_cache, 'TaxiNYC_C{}_P{}_T{}.h5'.format(
        len_c, len_p, len_t))
    if os.path.exists(fname) and CACHEDATA:
        X_train, Y_train, \
        X_test, Y_test, mmn, external_dim, \
        timestamp_train, timestamp_test = read_cache(
            fname, 'preprocessing_taxinyc.pkl')
        print("load %s successfully" % fname)
    else:
        X_train, Y_train, \
        X_test, Y_test, mmn, external_dim, \
        timestamp_train, timestamp_test = TaxiNYC.load_data(
            T=T, nb_flow=nb_flow, len_closeness=len_c, len_period=len_p, len_trend=len_t, len_test=len_test,
            preprocess_name='preprocessing_taxinyc.pkl', meta_data=True,
            meteorol_data=True, holiday_data=True, datapath=DATAPATH)
        if CACHEDATA:
            cache(fname, X_train, Y_train, X_test, Y_test,
                    external_dim, timestamp_train, timestamp_test)

    print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])
    print('=' * 10)

    # build model
    model = build_model('NY', len_c, len_p, len_t, nb_flow, map_height,
                        map_width, external_dim)
    
    model_fname = 'TaxiNYC1.c4.p4.t4.lr_0.00034.batchsize_16.best.h5'
    model.load_weights(os.path.join('../best_models', 'MST3D', model_fname))

    # evaluate and save results
    dict_multi_score = multi_step_2D(model, X_test, Y_test, mmn, len_c, step=5)

    for i in range(len(dict_multi_score)):
        csv_name = os.path.join('results', f'taxiny_step{i+1}.csv')
        save_to_csv(dict_multi_score[i], csv_name)
    

def bikenyc_evaluation():
    # parameters
    DATAPATH = '../data'
    CACHEDATA = True  # cache data or NOT
    T = 24  # number of time intervals in one day
    lr = 0.0002  # learning rate
    len_closeness = 4  # length of closeness dependent sequence
    len_period = 4  # length of peroid dependent sequence
    len_trend = 4  # length of trend dependent sequence

    nb_flow = 2
    days_test = 10
    len_test = T * days_test
    map_height, map_width = 16, 8

    path_cache = os.path.join(DATAPATH, 'CACHE', 'MST3D')
    if CACHEDATA and os.path.isdir(path_cache) is False:
        os.mkdir(path_cache)
    
    # load data
    print("loading data...")
    ts = time.time()
    fname = os.path.join(path_cache, 'BikeNYC_C{}_P{}_T{}.h5'.format(
        len_closeness, len_period, len_trend))
    if os.path.exists(fname) and CACHEDATA:
        X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = read_cache(
            fname, 'preprocessing_bikenyc.pkl')
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

    # build model
    model = build_model('NY', len_closeness, len_period, len_trend, nb_flow,
                        map_height, map_width, external_dim)
    model_fname = 'BikeNYC.c4.p4.t4.lr0.0002.best.h5'
    model.load_weights(os.path.join('../best_models', 'MST3D', model_fname))

    # evaluate and save results
    dict_multi_score = multi_step_2D(model, X_test, Y_test, mmn, len_closeness, step=5)

    for i in range(len(dict_multi_score)):
        csv_name = os.path.join('results', f'bikenyc_step{i+1}.csv')
        save_to_csv(dict_multi_score[i], csv_name)


if (__name__ == '__main__'):
    taxibj_evaluation()
    taxiny_evaluation()
    bikenyc_evaluation()
