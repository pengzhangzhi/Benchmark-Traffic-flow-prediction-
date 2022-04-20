# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import _pickle as pickle
import numpy as np
from star import *
from star.minmax_normalization import MinMaxNormalization
from star.config import Config
from star.STMatrix import STMatrix

np.random.seed(1337)  # for reproducibility

# parameters
# DATAPATH = Config().DATAPATH


def load_data(T=24, nb_flow=2, len_closeness=None, len_period=None, len_trend=None, len_test=None, len_val=None, preprocess_name='preprocessing.pkl', meta_data=True, datapath=None):
    assert(len_closeness + len_period + len_trend > 0)
    # load data
    data, timestamps = load_stdata(os.path.join(datapath, 'BikeNYC', 'NYC14_M16x8_T60_NewEnd.h5'))
    print(len(timestamps))
    # remove a certain day which does not have 48 timestamps
    data, timestamps = remove_incomplete_days(data, timestamps, T)
    # print(timestamps)
    # data = data[:, :nb_flow]
    data[data < 0] = 0.
    data_all = [data]
    timestamps_all = [timestamps]
    # minmax_scale
    data_train = data[:-len_test]
    print('train_data shape: ', data_train.shape)
    mmn = MinMaxNormalization()
    mmn.fit(data_train)
    data_all_mmn = []
    for d in data_all:
        data_all_mmn.append(mmn.transform(d))

    fpkl = open(preprocess_name, 'wb')
    for obj in [mmn]:
        pickle.dump(obj, fpkl)
    fpkl.close()

    XC, XP, XT = [], [], []
    XCPT = []
    Y = []
    timestamps_Y = []
    for data, timestamps in zip(data_all_mmn, timestamps_all):

        st = STMatrix(data, timestamps, T, CheckComplete=False)
        _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset(
            len_closeness=len_closeness, len_period=len_period, len_trend=len_trend)
        # _XCPT[:, 0:6, :, :] = _XC
        # _XCPT = np.zeros((_XC.shape[0], 2*(len_closeness+len_period+len_trend), 32, 32))
        print("_XC shape: ", _XC.shape, "_XP shape:", _XP.shape, "_XT shape:", _XT.shape)
        _XCPT = np.concatenate((_XC, _XP),axis=1)
        _XCPT = np.concatenate((_XCPT, _XT),axis=1)
        # _XCPT = np.concatenate((_XCPT, _XT),axis=1)
        # print(_XCPT.shape)
    # XC = np.vstack(XC)
    # XP = np.vstack(XP)
    # XT = np.vstack(XT)
        XCPT.append(_XCPT)
        Y.append(_Y)

        timestamps_Y += _timestamps_Y
        
    Y = np.vstack(Y)
    XCPT = np.vstack(XCPT)

    # print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)

    XCPT_train_all, Y_train_all = XCPT[:-len_test], Y[:-len_test]
    XCPT_train, Y_train = XCPT[:-len_val], Y[:-len_val]
    XCPT_val, Y_val = XCPT[-len_val:-len_test], Y[-len_val:-len_test]
    XCPT_test, Y_test = XCPT[-len_test:], Y[-len_test:]
    
    timestamp_train_all, timestamp_train, timestamp_val, timestamp_test = timestamps_Y[:-len_test], timestamps_Y[:-len_val], timestamps_Y[-len_val:-len_test], timestamps_Y[-len_test:]

    X_train_all, X_train, X_val, X_test = [], [], [], []

    X_train_all.append(XCPT_train_all)
    X_train.append(XCPT_train)
    X_val.append(XCPT_val)
    X_test.append(XCPT_test)

    meta_feature = []

    # for l, X_ in zip([len_closeness, len_period, len_trend], [XC_train, XP_train, XT_train]):
    #     if l > 0:
    #         X_train.append(X_)
    # for l, X_ in zip([len_closeness, len_period, len_trend], [XC_test, XP_test, XT_test]):
    #     if l > 0:
    #         X_test.append(X_)
    # print('train shape:', XC_train.shape, Y_train.shape, 'test shape: ', XC_test.shape, Y_test.shape)
    # load meta feature
    if meta_data:
        meta_feature = timestamp2vec(timestamps_Y)
        metadata_dim = meta_feature.shape[1]
        meta_feature_train_all, meta_feature_train, meta_feature_val, meta_feature_test = meta_feature[
        :-len_test], meta_feature[:-len_val], meta_feature[-len_val:-len_test], meta_feature[-len_test:]
        X_train_all.append(meta_feature_train_all)  
        X_train.append(meta_feature_train)
        X_val.append(meta_feature_val)
        X_test.append(meta_feature_test)
    else:
        metadata_dim = None
    for _X in X_train_all:
        print(_X.shape, )
    print()    
    for _X in X_train:
        print(_X.shape, )
    print()
    for _X in X_val:
        print(_X.shape, )
    print()
    for _X in X_test:
        print(_X.shape, )
    print()
    return X_train_all, Y_train_all, X_train, Y_train, X_val, Y_val, X_test, Y_test, mmn, metadata_dim, timestamp_train_all, timestamp_train, timestamp_val, timestamp_test

def load_data_kdd18(T=24, nb_flow=2, len_closeness=None, len_period=None, len_trend=None, len_test=None, len_val=None, preprocess_name='preprocessing.pkl', meta_data=True):
    assert(len_closeness + len_period + len_trend > 0)
    # load data
    data, timestamps = load_stdata(os.path.join(DATAPATH, 'BikeNYC', 'NYC14_M16x8_T60_NewEnd.h5'))
    print(len(timestamps))
    # remove a certain day which does not have 48 timestamps
    data, timestamps = remove_incomplete_days(data, timestamps, T)
    # print(timestamps)
    # data = data[:, :nb_flow]
    data[data < 0] = 0.
    data_all = [data]
    timestamps_all = [timestamps]
    # minmax_scale
    data_train = data[:-len_test]
    print('train_data shape: ', data_train.shape)
    mmn = MinMaxNormalization()
    mmn.fit(data_train)
    data_all_mmn = []
    for d in data_all:
        data_all_mmn.append(mmn.transform(d))

    fpkl = open('preprocessing.pkl', 'wb')
    for obj in [mmn]:
        pickle.dump(obj, fpkl)
    fpkl.close()

    XC, XP, XT = [], [], []
    XCPT = []
    Y = []
    timestamps_Y = []
    for data, timestamps in zip(data_all_mmn, timestamps_all):

        st = STMatrix(data, timestamps, T, CheckComplete=False)
        _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset(
            len_closeness=len_closeness, len_period=len_period, len_trend=len_trend)
        # _XCPT[:, 0:6, :, :] = _XC
        # _XCPT = np.zeros((_XC.shape[0], 2*(len_closeness+len_period+len_trend), 32, 32))
        print("_XC shape: ", _XC.shape, "_XP shape:", _XP.shape, "_XT shape:", _XT.shape)
        _XC = np.reshape(_XC,(_XP.shape[0],len_closeness,2,16,8))
        _XP = np.reshape(_XP,(_XP.shape[0],len_period*2,2,16,8))
        _XT = np.reshape(_XT,(_XT.shape[0],len_trend*2,2,16,8))

        _XCPT = np.concatenate((_XC, _XP),axis=1)
        _XCPT = np.concatenate((_XCPT, _XT),axis=1)
        _Y = np.expand_dims(_Y, axis=1)

        # _XCPT = np.concatenate((_XCPT, _XT),axis=1)
        # print(_XCPT.shape)
    # XC = np.vstack(XC)
    # XP = np.vstack(XP)
    # XT = np.vstack(XT)
        XCPT.append(_XCPT)
        Y.append(_Y)

        timestamps_Y += _timestamps_Y
        
    Y = np.vstack(Y)
    Y = Y.transpose(0, 2, 1, 3, 4)
    XCPT = np.vstack(XCPT)
    XCPT = XCPT.transpose(0, 2, 1, 3, 4)
    # print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)

    XCPT_train_all, Y_train_all = XCPT[:-len_test], Y[:-len_test]
    XCPT_train, Y_train = XCPT[:-len_val], Y[:-len_val]
    XCPT_val, Y_val = XCPT[-len_val:-len_test], Y[-len_val:-len_test]
    XCPT_test, Y_test = XCPT[-len_test:], Y[-len_test:]
    
    timestamp_train_all, timestamp_train, timestamp_val, timestamp_test = timestamps_Y[:-len_test], timestamps_Y[:-len_val], timestamps_Y[-len_val:-len_test], timestamps_Y[-len_test:]

    X_train_all, X_train, X_val, X_test = [], [], [], []

    X_train_all.append(XCPT_train_all)
    X_train.append(XCPT_train)
    X_val.append(XCPT_val)
    X_test.append(XCPT_test)

    meta_feature = []

    # for l, X_ in zip([len_closeness, len_period, len_trend], [XC_train, XP_train, XT_train]):
    #     if l > 0:
    #         X_train.append(X_)
    # for l, X_ in zip([len_closeness, len_period, len_trend], [XC_test, XP_test, XT_test]):
    #     if l > 0:
    #         X_test.append(X_)
    # print('train shape:', XC_train.shape, Y_train.shape, 'test shape: ', XC_test.shape, Y_test.shape)
    # load meta feature
    if meta_data:
        meta_feature = timestamp2vec(timestamps_Y)
        metadata_dim = meta_feature.shape[1]
        meta_feature_train_all, meta_feature_train, meta_feature_val, meta_feature_test = meta_feature[
        :-len_test], meta_feature[:-len_val], meta_feature[-len_val:-len_test], meta_feature[-len_test:]
        X_train_all.append(meta_feature_train_all)  
        X_train.append(meta_feature_train)
        X_val.append(meta_feature_val)
        X_test.append(meta_feature_test)
    else:
        metadata_dim = None
    for _X in X_train_all:
        print(_X.shape, )
    print()    
    for _X in X_train:
        print(_X.shape, )
    print()
    for _X in X_val:
        print(_X.shape, )
    print()
    for _X in X_test:
        print(_X.shape, )
    print()
    return X_train_all, Y_train_all, X_train, Y_train, X_val, Y_val, X_test, Y_test, mmn, metadata_dim, timestamp_train_all, timestamp_train, timestamp_val, timestamp_test
