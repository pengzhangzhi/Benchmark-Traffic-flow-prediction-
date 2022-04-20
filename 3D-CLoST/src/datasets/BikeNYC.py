# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import pickle
import numpy as np

from . import load_stdata
from ..preprocessing.minmax_normalization import MinMaxNormalization_01
from ..preprocessing import remove_incomplete_days
from ..utils import create_dict, create_mask
from .STMatrix import STMatrix
from ..preprocessing import timestamp2vec
# np.random.seed(1337)  # for reproducibility

# parameters
# DATAPATH = Config().DATAPATH


def load_data(T=24, nb_flow=2, len_closeness=None, len_period=None, len_trend=None, len_test=None, len_val=None, preprocess_name='preprocessing.pkl', meta_data=True, datapath=None, add_half=False):
    assert(len_closeness + len_period + len_trend > 0)
    # load data
    data, timestamps = load_stdata(os.path.join(datapath, 'BikeNYC', 'NYC14_M16x8_T60_NewEnd.h5'))
    # print(timestamps)
    # remove a certain day which does not have 48 timestamps
    data, timestamps = remove_incomplete_days(data, timestamps, T)
    data = data[:, :nb_flow]
    data[data < 0] = 0.

    # create mask
    ny_dict = create_dict(data, timestamps)
    mask = create_mask('NY', ny_dict)
    mask = np.moveaxis(mask, 0, -1)

    data_all = [data]
    timestamps_all = [timestamps]
    # minmax_scale
    data_train = data[:-len_test]
    print('train_data shape: ', data_train.shape)
    mmn = MinMaxNormalization_01()
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
        # instance-based dataset --> sequences with format as (X, Y) where X is a sequence of images and Y is an image.
        st = STMatrix(data, timestamps, T, CheckComplete=False, add_half=add_half)
        _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset(len_closeness=len_closeness, len_period=len_period, len_trend=len_trend)
        _XCPT = _XC
        if (len_period > 0):
            _XCPT = np.concatenate((_XC, _XP),axis=1)
        if (len_trend > 0):
            _XCPT = np.concatenate((_XCPT, _XT),axis=1)
        
        XCPT.append(_XCPT)
        Y.append(_Y)
        timestamps_Y += _timestamps_Y

    Y = np.vstack(Y)
    XCPT = np.vstack(XCPT)

    # print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)

    XCPT_train_all, Y_train_all = XCPT[:-len_test], Y[:-len_test]
    XCPT_train, Y_train = XCPT_train_all[:-len_val], Y_train_all[:-len_val]
    XCPT_val, Y_val = XCPT_train_all[-len_val:-len_test], Y_train_all[-len_val:-len_test]
    XCPT_test, Y_test = XCPT[-len_test:], Y[-len_test:]
    
    timestamp_train_all, timestamp_train, timestamp_val, timestamp_test = timestamps_Y[:-len_test], timestamps_Y[:-len_val], timestamps_Y[-len_val:-len_test], timestamps_Y[-len_test:]

    X_train_all, X_train, X_val, X_test = [], [], [], []

    X_train_all.append(XCPT_train_all)
    X_train.append(XCPT_train)
    X_val.append(XCPT_val)
    X_test.append(XCPT_test)


    # load meta feature
    if meta_data:
        meta_feature = timestamp2vec(timestamps_Y)
        metadata_dim = meta_feature.shape[1]
        meta_feature_train_all =  meta_feature[:-len_test]
        meta_feature_train, meta_feature_val, meta_feature_test = meta_feature_train_all[:-len_val], meta_feature[-len_val:-len_test], meta_feature[-len_test:]
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
    return X_train_all, Y_train_all, X_train, Y_train, X_val, Y_val, X_test, Y_test, mmn, metadata_dim, timestamp_train_all, timestamp_train, timestamp_val, timestamp_test, mask
