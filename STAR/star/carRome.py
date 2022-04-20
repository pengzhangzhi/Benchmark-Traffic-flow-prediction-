# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import _pickle as pickle
import numpy as np
import h5py
from star import *
from star.minmax_normalization import MinMaxNormalization
from star.config import Config
from star.STMatrix import STMatrix


def load_meteorol(timeslots, datapath):
    fname=os.path.join(datapath, 'Roma', 'Rome_Meteorology.h5')
    f = h5py.File(fname, 'r')
    Timeslot = f['date'].value
    WindSpeed = f['WindSpeed'].value
    Weather = f['Weather'].value
    Temperature = f['Temperature'].value
    f.close()

    M = dict()  # map timeslot to index
    for i, slot in enumerate(Timeslot):
        M[slot] = i

    WS = []  # WindSpeed
    WR = []  # Weather
    TE = []  # Temperature
    for slot in timeslots:
        predicted_id = M[slot]
        cur_id = predicted_id - 1
        WS.append(WindSpeed[cur_id])
        WR.append(Weather[cur_id])
        TE.append(Temperature[cur_id])

    WS = np.asarray(WS)
    WR = np.asarray(WR)
    TE = np.asarray(TE)

    # 0-1 scale
    WS = 1. * (WS - WS.min()) / (WS.max() - WS.min())
    TE = 1. * (TE - TE.min()) / (TE.max() - TE.min())

    print("shape: ", WS.shape, WR.shape, TE.shape)

    # concatenate all these attributes
    merge_data = np.hstack([WR, WS[:, None], TE[:, None]])

    # print('meger shape:', merge_data.shape)
    return merge_data

def load_holiday(timeslots, datapath):
    fname=os.path.join(datapath, 'Roma', 'Rome_Holiday.txt')
    f = open(fname, 'r')
    holidays = f.readlines()
    holidays = set([h.strip() for h in holidays])
    H = np.zeros(len(timeslots))
    for i, slot in enumerate(timeslots):
        if slot[:8] in holidays:
            H[i] = 1
    print(H.sum())
    # print(timeslots[H==1])
    return H[:, None]

def load_data(T=24*2, nb_flow=2, len_closeness=None, len_period=None, len_trend=None,
              len_test=None, len_val=None, preprocess_name='preprocessing.pkl',
              meta_data=True, meteorol_data=False, holiday_data=True, datapath=None, shape=(32,32)):
    assert(len_closeness + len_period + len_trend > 0)
    # load data
    if (shape == (32,32)):
        data, timestamps = load_stdata(os.path.join(datapath, 'Roma', 'AllMap', 'Roma_32x32_30_minuti_north.h5'))
    else:
        data, timestamps = load_stdata(os.path.join(datapath, 'Roma', 'AllMap', 'Roma_16x8_1_ora_resize_north.h5'))
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
        time_feature = timestamp2vec(timestamps_Y)
        meta_feature.append(time_feature)
        if holiday_data:
            # load holiday
            holiday_feature = load_holiday(timestamps_Y, datapath)
            meta_feature.append(holiday_feature)
        if meteorol_data:
            # load meteorol data
            meteorol_feature = load_meteorol(timestamps_Y, datapath)
            meta_feature.append(meteorol_feature)
        
        meta_feature = np.hstack(meta_feature) if len(
            meta_feature) > 0 else np.asarray(meta_feature)
        metadata_dim = meta_feature.shape[1] if len(
            meta_feature.shape) > 1 else None
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
