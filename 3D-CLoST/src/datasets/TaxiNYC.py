from __future__ import print_function

import os
#import cPickle as pickle
import pickle
from copy import copy
import numpy as np
import h5py

from ..utils import create_dict, create_mask
from . import load_stdata, stat
from ..preprocessing import timestamp2vec
from ..preprocessing.minmax_normalization import MinMaxNormalization_01
from .STMatrix import STMatrix

def load_holiday(timeslots, datapath):
    fname=os.path.join(datapath, 'TaxiNYC', 'NY_Holiday.txt')
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


def load_meteorol(timeslots, datapath):
    '''
    timeslots: the predicted timeslots
    In real-world, we dont have the meteorol data in the predicted timeslot, instead,
    we use the meteoral at previous timeslots, i.e., slot = predicted_slot - timeslot (you can use predicted meteorol data as well)
    '''
    def adjust_timeslots(timeslot):
        timeslot_str = timeslot.decode("utf-8")
        interval = timeslot_str[-2:]
        new_interval = f'{int(interval)+1:02}'
        return bytes(timeslot_str[:-2] + new_interval, encoding='utf8')
    timeslots = [adjust_timeslots(t) for t in timeslots]

    fname=os.path.join(datapath, 'TaxiNYC', 'NY_Meteorology.h5')
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

def remove_incomplete_days(data, timestamps, T=24):
    # remove a certain day which has not T timestamps
    days = []  # available days: some day only contain some seqs
    days_incomplete = []
    i = 0
    while i < len(timestamps):
        if int(timestamps[i][8:]) != 0:
            i += 1
        elif i+T-1 < len(timestamps) and int(timestamps[i+T-1][8:]) == T-1:
            days.append(timestamps[i][:8])
            i += T
        else:
            days_incomplete.append(timestamps[i][:8])
            i += 1
    print("incomplete days: ", days_incomplete)
    days = set(days)
    idx = []
    for i, t in enumerate(timestamps):
        if t[:8] in days:
            idx.append(i)

    data = data[idx]
    timestamps = [timestamps[i] for i in idx]
    return data, timestamps

def load_data(T=24, nb_flow=2, len_closeness=None, len_period=None, len_trend=None,
              len_test=None, len_val=None, preprocess_name='preprocessing_taxinyc.pkl',
              meta_data=True, meteorol_data=False, holiday_data=False, datapath=None, add_half=False):
    assert(len_closeness + len_period + len_trend > 0)
    # load data
    # 10 - 14
    data_all = []
    timestamps_all = list()
    for year in range(10, 15):
        fname = os.path.join(
            datapath, 'TaxiNYC', 'NYC{}_Taxi_M16x8_T60_InOut.h5'.format(year))
        print("file name: ", fname)
        stat(fname)
        data, timestamps = load_stdata(fname)
        # print(timestamps)
        # remove a certain day which does not have 48 timestamps
        data, timestamps = remove_incomplete_days(data, timestamps, T)
        data = data[:, :nb_flow]
        data[data < 0] = 0.
        data_all.append(data)
        timestamps_all.append(timestamps)
        print("\n")
    
    # create mask
    ny_dict = create_dict(
        np.vstack(copy(data_all)), # all data in one array
        [item for sublist in timestamps_all for item in sublist] # all timestamps in one list
    )
    mask = create_mask('NY', ny_dict)
    mask = np.moveaxis(mask, 0, -1)

    # minmax_scale
    data_train = np.vstack(copy(data_all))[:-len_test]
    print('train_data shape: ', data_train.shape)
    mmn = MinMaxNormalization_01()
    mmn.fit(data_train)
    data_all_mmn = [mmn.transform(d) for d in data_all]

    fpkl = open(preprocess_name, 'wb')
    for obj in [mmn]:
        pickle.dump(obj, fpkl)
    fpkl.close()
    XCPT = []
    Y = []
    timestamps_Y = []
    for data, timestamps in zip(data_all_mmn, timestamps_all):
        # instance-based dataset --> sequences with format as (X, Y) where X is
        # a sequence of images and Y is an image.
        st = STMatrix(data, timestamps, T, CheckComplete=False, Hours0_23=True, add_half=add_half)
        _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset(
            len_closeness=len_closeness, len_period=len_period, len_trend=len_trend)

        _XCPT = _XC
        if (len_period > 0):
            _XCPT = np.concatenate((_XC, _XP),axis=1)
        if (len_trend > 0):
            _XCPT = np.concatenate((_XCPT, _XT),axis=1)

        XCPT.append(_XCPT)

        Y.append(_Y)
        # _Y=[]
        timestamps_Y += _timestamps_Y

    
    XCPT = np.vstack(XCPT)
    Y = np.vstack(Y)

    print("XCPT shape: ", XCPT.shape, "Y shape:", Y.shape)

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

    print('train shape:', XCPT_train_all.shape, Y_train_all.shape,
          'train shape:', XCPT_train.shape, Y_train.shape,
          'test shape: ', XCPT_val.shape, Y_val.shape,
          'test shape: ', XCPT_test.shape, Y_test.shape)
    
    # load meta feature
    meta_feature = []
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
