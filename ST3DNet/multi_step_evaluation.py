from ST3DNet import *
import pickle
from utils import *
import os
import math
import numpy as np
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam

from evaluation import evaluate
from multi_step import multi_step_2D

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
    T = 48  # number of time intervals in one day
    lr = 0.0001  # learning rate
    len_closeness = 6  # length of closeness dependent sequence
    len_period = 0  # length of peroid dependent sequence
    len_trend = 2  # length of trend dependent sequence
    nb_residual_unit = 7   # number of residual units
    nb_flow = 2  # there are two types of flows: new-flow and end-flow
    days_test = 7*4  
    len_test = T * days_test
    map_height, map_width = 32, 32  # grid size
    m_factor = 1

    filename = os.path.join("../data", 'CACHE', 'ST3DNet', 'TaxiBJ_c%d_p%d_t%d_noext'%(len_closeness, len_period, len_trend))
    f = open(filename, 'rb')
    X_train = pickle.load(f)
    Y_train = pickle.load(f)
    X_test = pickle.load(f)
    Y_test = pickle.load(f)
    mmn = pickle.load(f)
    external_dim = pickle.load(f)
    timestamp_train = pickle.load(f)
    timestamp_test = pickle.load(f)

    for i in X_train:
        print(i.shape)

    Y_train = mmn.inverse_transform(Y_train)  # X is MaxMinNormalized, Y is real value
    Y_test = mmn.inverse_transform(Y_test)

    c_conf = (len_closeness, nb_flow, map_height,
                  map_width) if len_closeness > 0 else None
    t_conf = (len_trend, nb_flow, map_height,
              map_width) if len_trend > 0 else None

    # build model
    model = ST3DNet(c_conf=c_conf, t_conf=t_conf, external_dim=external_dim, nb_residual_unit=nb_residual_unit)
    model_fname = 'TaxiBJ.c6.p0.t2.resunit7.lr0.0001.cont.best.h5'
    model.load_weights(os.path.join('../best_models', 'ST3DNet', model_fname))

    # evaluate and save results
    dict_multi_score = multi_step_2D(model, X_test, Y_test, mmn, len_closeness, external_dim, step=5)

    for i in range(len(dict_multi_score)):
        csv_name = os.path.join('results', f'taxibj_step{i+1}.csv')
        save_to_csv(dict_multi_score[i], csv_name)


def taxiny_evaluation():
    # params
    T = 24  # number of time intervals in one day
    len_closeness = len_c =  6  # length of closeness dependent sequence
    len_period = len_p = 0  # length of peroid dependent sequence
    len_trend = len_t = 4  # length of trend dependent sequence
    nb_residual_unit = 5   # number of residual units
    nb_flow = 2  # there are two types of flows: new-flow and end-flow
    days_test = 7*4  
    len_test = T * days_test
    map_height, map_width = 16, 8  # grid size

    filename = os.path.join("../data", 'CACHE', 'ST3DNet', 'TaxiNYC_c%d_p%d_t%d_noext'%(len_closeness, len_period, len_trend))
    f = open(filename, 'rb')
    X_train = pickle.load(f)
    Y_train = pickle.load(f)
    X_test = pickle.load(f)
    Y_test = pickle.load(f)
    mmn = pickle.load(f)
    external_dim = pickle.load(f)
    timestamp_train = pickle.load(f)
    timestamp_test = pickle.load(f)

    for i in X_train:
        print(i.shape)

    Y_train = mmn.inverse_transform(Y_train)  # X is MaxMinNormalized, Y is real value
    Y_test = mmn.inverse_transform(Y_test)

    c_conf = (len_closeness, nb_flow, map_height,
                  map_width) if len_closeness > 0 else None
    t_conf = (len_trend, nb_flow, map_height,
              map_width) if len_trend > 0 else None

    # build model
    model = ST3DNet(c_conf=c_conf, t_conf=t_conf, external_dim=external_dim,
                    nb_residual_unit=nb_residual_unit)

    model_fname = 'TaxiNYC2.c6.p0.t4.resunits_5.lr_0.00095.batchsize_16.best.h5'
    model.load_weights(os.path.join('../best_models', 'ST3DNet', model_fname))

    # evaluate and save results
    dict_multi_score = multi_step_2D(model, X_test, Y_test, mmn, len_closeness, external_dim, step=5)

    for i in range(len(dict_multi_score)):
        csv_name = os.path.join('results', f'taxiny_step{i+1}.csv')
        save_to_csv(dict_multi_score[i], csv_name)
    

def bikenyc_evaluation():
    T = 24  # number of time intervals in one day
    len_closeness = 6  # length of closeness dependent sequence
    len_period = 0  # length of peroid dependent sequence
    len_trend = 4  # length of trend dependent sequence
    nb_residual_unit = 4   # number of residual units
    nb_flow = 2  # there are two types of flows: new-flow and end-flow
    days_test = 10  # divide data into two subsets: Train & Test, of which the test set is the last 10 days
    len_test = T * days_test
    map_height, map_width = 16, 8  # grid size

    filename = os.path.join("../data", 'CACHE', 'ST3DNet', 'BikeNYC_c%d_p%d_t%d_noext'%(len_closeness, len_period, len_trend))
    f = open(filename, 'rb')
    X_train = pickle.load(f)
    Y_train = pickle.load(f)
    X_test = pickle.load(f)
    Y_test = pickle.load(f)
    mmn = pickle.load(f)
    external_dim = pickle.load(f)
    timestamp_train = pickle.load(f)
    timestamp_test = pickle.load(f)

    for i in X_train:
        print(i.shape)

    Y_train = mmn.inverse_transform(Y_train)  # X is MaxMinNormalized, Y is real value
    Y_test = mmn.inverse_transform(Y_test)

    c_conf = (len_closeness, nb_flow, map_height,
                  map_width) if len_closeness > 0 else None
    t_conf = (len_trend, nb_flow, map_height,
              map_width) if len_trend > 0 else None

    # build model
    model = ST3DNet(c_conf=c_conf, t_conf=t_conf, external_dim=external_dim,
                    nb_residual_unit=nb_residual_unit)
    
    model_fname = 'BikeNYC.c6.p0.t4.resunit4.lr2e-05.cont.best.h5'
    model.load_weights(os.path.join('../best_models', 'ST3DNet', model_fname))

    # evaluate and save results
    dict_multi_score = multi_step_2D(model, X_test, Y_test, mmn, len_closeness, external_dim, step=5)

    for i in range(len(dict_multi_score)):
        csv_name = os.path.join('results', f'bikenyc_step{i+1}.csv')
        save_to_csv(dict_multi_score[i], csv_name)


if (__name__ == '__main__'):
    taxibj_evaluation()
    taxiny_evaluation()
    bikenyc_evaluation()
