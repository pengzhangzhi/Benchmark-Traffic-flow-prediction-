from __future__ import print_function
import os
import _pickle as pickle
import numpy as np
import math
import h5py

import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam

from src.model import build_model
import src.metrics as metrics
from src.datasets import TaxiBJ, TaxiNYC, BikeNYC
from src.evaluation import evaluate
from src.multi_step import multi_step_2D
from cache_utils import cache, read_cache


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


def taxibj_evaluation():
    # parameters
    DATAPATH = '../data' 
    T = 48  # number of time intervals in one day
    CACHEDATA = True  # cache data or NOT

    lr = 0.001  # learning rate
    len_c = 4  # length of closeness dependent sequence
    len_p = 1  # length of peroid dependent sequence
    len_t = 0  # length of trend dependent sequence

    nb_flow = 2  # there are two types of flows: new-flow and end-flow
    # divide data into two subsets: Train & Test
    days_test = 7*4
    len_test = T*days_test
    len_val = 2*len_test

    map_height, map_width = 32, 32  # grid size

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
    preprocess_name = 'preprocessing_taxibj.pkl'
    fname = os.path.join(path_cache, 'TaxiBJ_C{}_P{}_T{}.h5'.format(
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
        timestamp_train_all, timestamp_train, timestamp_val, timestamp_test, mask = TaxiBJ.load_data(
            T=T, nb_flow=nb_flow, len_closeness=len_c, len_period=len_p, len_trend=len_t, len_test=len_test,
            len_val=len_val, preprocess_name=preprocess_name, meta_data=True, datapath=DATAPATH, add_half=True)
        if CACHEDATA:
            cache(fname, X_train_all, Y_train_all, X_train, Y_train, X_val, Y_val, X_test, Y_test,
                    external_dim, timestamp_train_all, timestamp_train, timestamp_val, timestamp_test, mask)

    # build model
    model = build_model('BJ', X_train,  Y_train,  conv_filt=64, kernel_sz=(3,3,3), 
                        mask=mask, lstm=500, lstm_number=2, add_external_info=True,
                        lr = 0.001, save_model_pic=None)
    model_fname = 'TaxiBJ.c4.p1.t0.iter7.best.h5'
    model.load_weights(os.path.join('../best_models', '3DCLoST', model_fname))

    # evaluate
    dict_multi_score = multi_step_2D(model, X_test, Y_test, mmn, len_c, step=5)

    for i in range(len(dict_multi_score)):
        csv_name = os.path.join('results', f'taxibj_step{i+1}.csv')
        save_to_csv(dict_multi_score[i], csv_name)


def taxiny_evaluation():
    DATAPATH = '../data'
    T = 24  # number of time intervals in one day
    CACHEDATA = True  # cache data or NOT

    len_closeness = len_c = 2  # length of closeness dependent sequence
    len_period = len_p = 0  # length of peroid dependent sequence
    len_trend = len_t = 1  # length of trend dependent sequence
    lr = 0.0015
    lstm = 500
    lstm_number = 2

    nb_flow = 2
    days_test = 7*4
    len_test = T*days_test
    len_val = 2*len_test

    map_height, map_width = 16, 8  # grid size

    path_cache = os.path.join(DATAPATH, 'CACHE', '3D-CLoST')  # cache path
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

    # build model
    model = build_model('NY', X_train,  Y_train, conv_filt=64, kernel_sz=(2,3,3), 
                        mask=mask, lstm=lstm, lstm_number=lstm_number, add_external_info=True,
                        lr=lr, save_model_pic=None)
    
    model_fname = 'TaxiNYC3.c2.p0.t1.lstm_500.lstmnumber_2.lr_0.00076.batchsize_16.best.h5'
    model.load_weights(os.path.join('../best_models', '3DCLoST', model_fname))

    # evaluate and save results
    dict_multi_score = multi_step_2D(model, X_test, Y_test, mmn, len_c, step=5)

    for i in range(len(dict_multi_score)):
        csv_name = os.path.join('results', f'taxiny_step{i+1}.csv')
        save_to_csv(dict_multi_score[i], csv_name)
    

def bikenyc_evaluation():
    DATAPATH = '../data'
    T = 24  # number of time intervals in one day
    CACHEDATA = True  # cache data or NOT

    lr = 0.001  # learning rate
    len_c = 2  # length of closeness dependent sequence
    len_p = 0  # length of peroid dependent sequence
    len_t = 1  # length of trend dependent sequence

    nb_flow = 2
    days_test = 10
    len_test = T*days_test
    len_val = 2*len_test

    map_height, map_width = 16, 8

    path_cache = os.path.join(DATAPATH, 'CACHE', '3D-CLoST')  # cache path
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

    # build model
    model = build_model('NY', X_train,  Y_train, conv_filt=64, kernel_sz=(2,3,3), 
                        mask=mask, lstm=350, lstm_number=2, add_external_info=True,
                        lr=0.001, save_model_pic=None)
    
    model_fname = 'BikeNYC.c2.p0.t1.iter7.best.h5'
    model.load_weights(os.path.join('../best_models', '3DCLoST', model_fname))

    # evaluate and save results
    dict_multi_score = multi_step_2D(model, X_test, Y_test, mmn, len_c, step=5)

    for i in range(len(dict_multi_score)):
        csv_name = os.path.join('results', f'bikenyc_step{i+1}.csv')
        save_to_csv(dict_multi_score[i], csv_name)


if (__name__ == '__main__'):
    taxibj_evaluation()
    taxiny_evaluation()
    bikenyc_evaluation()
