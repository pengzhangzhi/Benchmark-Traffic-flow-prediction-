import numpy as np
import time
import os
import json
import pickle as pickle
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import tensorflow as tf
from keras import backend as K

from utils import cache, read_cache
from src import TaxiBJ3d, TaxiNYC3d, BikeNYC3d
from src.evaluation import evaluate
from src.streednet import build_model
from src.multi_step import multi_step_2D


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

    len_closeness = 4  # length of closeness dependent sequence
    len_period = 0  # length of peroid dependent sequence
    len_trend = 0  # length of trend dependent sequence

    nb_flow = 2  # there are two types of flows: new-flow and end-flow
    days_test = 4*7 # 4 weeks
    len_test = T * days_test
    len_val = 2 * len_test

    map_height, map_width = 32, 32  # grid size

    cache_folder = 'Autoencoder/model3'
    path_cache = os.path.join(DATAPATH, 'CACHE', cache_folder)  # cache path
    if CACHEDATA and os.path.isdir(path_cache) is False:
        os.mkdir(path_cache)

    # load data
    print("loading data...")
    fname = os.path.join(path_cache, 'TaxiBJ_withMeteo_C{}_P{}_T{}.h5'.format(
        len_closeness, len_period, len_trend))
    if os.path.exists(fname) and CACHEDATA:
        X_train_all, Y_train_all, X_train, Y_train, \
        X_val, Y_val, X_test, Y_test, mmn, external_dim, \
        timestamp_train_all, timestamp_train, timestamp_val, timestamp_test = read_cache(
            fname, 'preprocessing_bj.pkl')
        print("load %s successfully" % fname)
    else:
        X_train_all, Y_train_all, X_train, Y_train, \
        X_val, Y_val, X_test, Y_test, mmn, external_dim, \
        timestamp_train_all, timestamp_train, timestamp_val, timestamp_test = TaxiBJ3d.load_data(
            T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,
            len_val=len_val, preprocess_name='preprocessing_bj.pkl', meta_data=True, meteorol_data=True, holiday_data=True, datapath=DATAPATH)
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

    model_fname = 'model3resunit_doppia_attention.TaxiBJ9.c4.p0.t0.encoderblocks_3.kernel_size_3.lr_0.0007.batchsize_16.best.h5'
    model.load_weights(os.path.join('../best_models', 'model3', model_fname))

    # evaluate and save results
    dict_multi_score = multi_step_2D(model, X_test, Y_test, mmn, len_closeness, step=5)

    for i in range(len(dict_multi_score)):
        csv_name = os.path.join('results', f'taxibj_step{i+1}.csv')
        save_to_csv(dict_multi_score[i], csv_name)


def taxiny_evaluation():
    DATAPATH = '../data'
    T = 24  # number of time intervals in one day
    CACHEDATA = True  # cache data or NOT

    len_closeness = 4  # length of closeness dependent sequence
    len_period = 0  # length of peroid dependent sequence
    len_trend = 0  # length of trend dependent sequence

    nb_flow = 2  # there are two types of flows: new-flow and end-flow
    days_test = 4*7 # 4 weeks
    len_test = T * days_test
    len_val = 2 * len_test

    map_height, map_width = 16, 8  # grid size

    cache_folder = 'Autoencoder/model3'
    path_cache = os.path.join(DATAPATH, 'CACHE', cache_folder)  # cache path
    if CACHEDATA and os.path.isdir(path_cache) is False:
        os.mkdir(path_cache)

    # load data
    print("loading data...")
    fname = os.path.join(path_cache, 'TaxiNYC_C{}_P{}_T{}.h5'.format(
        len_closeness, len_period, len_trend))
    if os.path.exists(fname) and CACHEDATA:
        X_train_all, Y_train_all, X_train, Y_train, \
        X_val, Y_val, X_test, Y_test, mmn, external_dim, \
        timestamp_train_all, timestamp_train, timestamp_val, timestamp_test = read_cache(
            fname, 'preprocessing_nyc.pkl')
        print("load %s successfully" % fname)
    else:
        X_train_all, Y_train_all, X_train, Y_train, \
        X_val, Y_val, X_test, Y_test, mmn, external_dim, \
        timestamp_train_all, timestamp_train, timestamp_val, timestamp_test = TaxiNYC3d.load_data(
            T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,
            len_val=len_val, preprocess_name='preprocessing_nyc.pkl', meta_data=True, meteorol_data=True, holiday_data=True, datapath=DATAPATH)
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

    model_fname = 'model3resunit_doppia_attention.TaxiNYC0.c4.p0.t0.encoderblocks_2.kernel_size_3.lr_0.00086.batchsize_48.best.h5'
    model.load_weights(os.path.join('../best_models', 'model3', model_fname))

    # evaluate and save results
    dict_multi_score = multi_step_2D(model, X_test, Y_test, mmn, len_closeness, step=5)

    for i in range(len(dict_multi_score)):
        csv_name = os.path.join('results', f'taxiny_step{i+1}.csv')
        save_to_csv(dict_multi_score[i], csv_name)


def bikenyc_evaluation():
    DATAPATH = '../data'
    T = 24  # number of time intervals in one day
    CACHEDATA = True  # cache data or NOT

    len_closeness = 4  # length of closeness dependent sequence
    len_period = 0  # length of peroid dependent sequence
    len_trend = 0  # length of trend dependent sequence

    nb_flow = 2
    days_test = 10
    len_test = T * days_test
    len_val = 2 * len_test

    map_height, map_width = 16, 8

    cache_folder = 'Autoencoder/model3'
    path_cache = os.path.join(DATAPATH, 'CACHE', cache_folder)
    if CACHEDATA and os.path.isdir(path_cache) is False:
        os.mkdir(path_cache)

    # load data
    print("loading data...")
    fname = os.path.join(path_cache, 'BikeNYC_C{}_P{}_T{}.h5'.format(
        len_closeness, len_period, len_trend))
    if os.path.exists(fname) and CACHEDATA:
        X_train_all, Y_train_all, X_train, Y_train, \
        X_val, Y_val, X_test, Y_test, mmn, external_dim, \
        timestamp_train_all, timestamp_train, timestamp_val, timestamp_test = read_cache(
            fname, 'preprocessing_bikenyc.pkl')
        print("load %s successfully" % fname)
    else:
        X_train_all, Y_train_all, X_train, Y_train, \
        X_val, Y_val, X_test, Y_test, mmn, external_dim, \
        timestamp_train_all, timestamp_train, timestamp_val, timestamp_test = BikeNYC3d.load_data(
            T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend,
            len_test=len_test,
            len_val=len_val, preprocess_name='preprocessing_bikenyc.pkl', meta_data=True, datapath=DATAPATH)
        if CACHEDATA:
            cache(fname, X_train_all, Y_train_all, X_train, Y_train, X_val, Y_val, X_test, Y_test,
                  external_dim, timestamp_train_all, timestamp_train, timestamp_val, timestamp_test)

    # build model
    model = build_model(
        len_closeness, len_period, len_trend, nb_flow, map_height, map_width,
        external_dim=external_dim,
        encoder_blocks=2,
        filters=[64,64,64,16],
        kernel_size=3,
        num_res=2
    )

    model_fname = 'model3resunit_doppia_attention.BikeNYC6.c4.p0.t0.encoderblocks_2.kernel_size_3.lr_0.0001.batchsize_16.best2.h5'
    model.load_weights(os.path.join('../best_models', 'model3', model_fname))

    # evaluate and save results
    dict_multi_score = multi_step_2D(model, X_test, Y_test, mmn, len_closeness, step=5)

    for i in range(len(dict_multi_score)):
        csv_name = os.path.join('results', f'bikenyc_step{i+1}.csv')
        save_to_csv(dict_multi_score[i], csv_name)


if (__name__ == '__main__'):
    taxibj_evaluation()
    taxiny_evaluation()
    bikenyc_evaluation()
