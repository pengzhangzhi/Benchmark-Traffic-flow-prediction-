import pandas as pd
from datetime import datetime
import os
import six.moves.cPickle as pickle
import numpy as np
import h5py
import time
from utils import *
from copy import copy


def remove_incomplete_days(data, timestamps, T=48):
    # remove a certain day which has not 48 timestamps
    days = []  # available days: some day only contain some seqs
    days_incomplete = []
    i = 0
    while i < len(timestamps):
        if int(timestamps[i][8:]) != 1:
            i += 1
        elif i+T-1 < len(timestamps) and int(timestamps[i+T-1][8:]) == T:
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

def remove_incomplete_days_taxiNYC(data, timestamps, T=24):
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

def load_holiday_taxiNYC(timeslots, datapath):
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


def load_meteorol_taxiNYC(timeslots, datapath):
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

def load_stdata(fname):
    f = h5py.File(fname, 'r')
    data = f['data'].value
    timestamps = f['date'].value
    f.close()
    return data, timestamps


def string2timestamp(strings, T=48):
    '''
    strings: list, eg. ['2017080912','2017080913']
    return: list, eg. [Timestamp('2017-08-09 05:30:00'), Timestamp('2017-08-09 06:00:00')]
    '''
    timestamps = []
    time_per_slot = 24.0 / T
    num_per_T = T // 24
    for t in strings:
        year, month, day, slot = int(t[:4]), int(t[4:6]), int(t[6:8]), int(t[8:])-1
        timestamps.append(pd.Timestamp(datetime(year, month, day, hour=int(slot * time_per_slot), minute=(slot % num_per_T) * int(60.0 * time_per_slot))))

    return timestamps

def string2timestamp_taxiNYC(strings, T=48):
    timestamps = []

    time_per_slot = 24.0 / T
    num_per_T = T // 24
    for t in strings:
        year, month, day, slot = int(t[:4]), int(t[4:6]), int(t[6:8]), int(t[8:])
        timestamps.append(pd.Timestamp(datetime(year, month, day, hour=int(slot * time_per_slot), minute=(slot % num_per_T) * int(60.0 * time_per_slot))))

    return timestamps


class STMatrix(object):
    """docstring for STMatrix"""

    def __init__(self, data, timestamps, T=48, CheckComplete=True, Hours0_23=False):
        super(STMatrix, self).__init__()
        assert len(data) == len(timestamps)
        self.data = data
        self.data_1 = data[:, 0, :, :]
        self.data_2 = data[:, 1, :, :]
        self.timestamps = timestamps
        self.T = T
        func = string2timestamp_taxiNYC if Hours0_23 else string2timestamp
        self.pd_timestamps = func(timestamps, T=self.T)
        if CheckComplete:
            self.check_complete()
        # index
        self.make_index()

    def make_index(self):
        self.get_index = dict()
        for i, ts in enumerate(self.pd_timestamps):
            self.get_index[ts] = i

    def check_complete(self):
        missing_timestamps = []
        offset = pd.DateOffset(minutes=24 * 60 // self.T)
        pd_timestamps = self.pd_timestamps
        i = 1
        while i < len(pd_timestamps):
            if pd_timestamps[i - 1] + offset != pd_timestamps[i]:
                missing_timestamps.append("(%s -- %s)" % (pd_timestamps[i - 1], pd_timestamps[i]))
            i += 1
        for v in missing_timestamps:
            print(v)
        assert len(missing_timestamps) == 0

    def get_matrix(self, timestamp):
        return self.data[self.get_index[timestamp]]

    def get_matrix_1(self, timestamp):  # in_flow
        ori_matrix = self.data_1[self.get_index[timestamp]]
        new_matrix = ori_matrix[np.newaxis, :]
        # print("new_matrix shape:",new_matrix.shape) #(1, 32, 32)
        return new_matrix

    def get_matrix_2(self, timestamp):  # out_flow
        ori_matrix = self.data_2[self.get_index[timestamp]]
        new_matrix = ori_matrix[np.newaxis, :]
        # print("new_matrix shape:",new_matrix.shape) #(1, 32, 32)
        return new_matrix

    def save(self, fname):
        pass

    def check_it(self, depends):
        for d in depends:
            if d not in self.get_index.keys():
                return False
        return True

    def create_dataset_3D(self, len_closeness=3, len_trend=3, TrendInterval=7, len_period=3, PeriodInterval=1):
        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
        XC = []
        XP = []
        XT = []
        Y = []
        timestamps_Y = []
        depends = [range(1, len_closeness + 1),
                   [PeriodInterval * self.T * j for j in range(1, len_period + 1)],
                   [TrendInterval * self.T * j for j in range(1, len_trend + 1)]]

        i = max(self.T * TrendInterval * len_trend, self.T * PeriodInterval * len_period, len_closeness)
        while i < len(self.pd_timestamps):
            Flag = True
            for depend in depends:
                if Flag is False:
                    break
                Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame for j in depend])

            if Flag is False:
                i += 1
                continue

            # closeness
            c_1_depends = list(depends[0])  # in_flow
            c_1_depends.sort(reverse=True)
            # print('----- c_1_depends:',c_1_depends)

            c_2_depends = list(depends[0])  # out_flow
            c_2_depends.sort(reverse=True)
            # print('----- c_2_depends:',c_2_depends)

            x_c_1 = [self.get_matrix_1(self.pd_timestamps[i] - j * offset_frame) for j in
                     c_1_depends]  # [(1,32,32),(1,32,32),(1,32,32)] in
            x_c_2 = [self.get_matrix_2(self.pd_timestamps[i] - j * offset_frame) for j in
                     c_2_depends]  # [(1,32,32),(1,32,32),(1,32,32)] out

            x_c_1_all = np.vstack(x_c_1)  # x_c_1_all.shape  (3, 32, 32)
            x_c_2_all = np.vstack(x_c_2)  # x_c_1_all.shape  (3, 32, 32)

            x_c_1_new = x_c_1_all[np.newaxis, :]  # (1, 3, 32, 32)
            x_c_2_new = x_c_2_all[np.newaxis, :]  # (1, 3, 32, 32)

            x_c = np.vstack([x_c_1_new, x_c_2_new])  # (2, 3, 32, 32)

            # period
            p_depends = list(depends[1])
            if (len(p_depends) > 0):
                p_depends.sort(reverse=True)
                # print('----- p_depends:',p_depends)

                x_p_1 = [self.get_matrix_1(self.pd_timestamps[i] - j * offset_frame) for j in p_depends]
                x_p_2 = [self.get_matrix_2(self.pd_timestamps[i] - j * offset_frame) for j in p_depends]

                x_p_1_all = np.vstack(x_p_1)  # [(3,32,32),(3,32,32),...]
                x_p_2_all = np.vstack(x_p_2)  # [(3,32,32),(3,32,32),...]

                x_p_1_new = x_p_1_all[np.newaxis, :]  # (1, 3, 32, 32)
                x_p_2_new = x_p_2_all[np.newaxis, :]  # (1, 3, 32, 32)

                x_p = np.vstack([x_p_1_new, x_p_2_new])  # (2, 3, 32, 32)

            # trend
            t_depends = list(depends[2])
            if (len(t_depends) > 0):
                t_depends.sort(reverse=True)

                x_t_1 = [self.get_matrix_1(self.pd_timestamps[i] - j * offset_frame) for j in t_depends]
                x_t_2 = [self.get_matrix_2(self.pd_timestamps[i] - j * offset_frame) for j in t_depends]

                x_t_1_all = np.vstack(x_t_1)  # [(3,32,32),(3,32,32),...]
                x_t_2_all = np.vstack(x_t_2)  # [(3,32,32),(3,32,32),...]

                x_t_1_new = x_t_1_all[np.newaxis, :]  # (1, 3, 32, 32)
                x_t_2_new = x_t_2_all[np.newaxis, :]  # (1, 3, 32, 32)

                x_t = np.vstack([x_t_1_new, x_t_2_new])  # (2, 3, 32, 32)

            y = self.get_matrix(self.pd_timestamps[i])

            if len_closeness > 0:
                XC.append(x_c)
            if len_period > 0:
                XP.append(x_p)
            if len_trend > 0:
                XT.append(x_t)
            Y.append(y)
            timestamps_Y.append(self.timestamps[i])
            i += 1

        XC = np.asarray(XC)
        XP = np.asarray(XP)
        XT = np.asarray(XT)
        Y = np.asarray(Y)
        print("3D matrix - XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
        return XC, XP, XT, Y, timestamps_Y


def timestamp2vec(timestamps):
    # tm_wday range [0, 6], Monday is 0
    vec = [time.strptime(str(t[:8], encoding='utf-8'), '%Y%m%d').tm_wday for t in timestamps]  # python3
    ret = []
    for i in vec:
        v = [0 for _ in range(7)]
        v[i] = 1
        if i >= 5:
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday
        ret.append(v)
    return np.asarray(ret)

def load_holiday(timeslots, datapath):
    fname = os.path.join(datapath, 'TaxiBJ', 'BJ_Holiday.txt')
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

def load_meteorol_rome(timeslots, datapath):
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

def load_holiday_rome(timeslots, datapath):
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

def load_meteorol_rome2(timeslots, datapath):
    fname=os.path.join(datapath, 'Roma', 'Rome_Meteorology_2.h5')
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

def load_holiday_rome2(timeslots, datapath):
    fname=os.path.join(datapath, 'Roma', 'Rome_Holiday_2.txt')
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
    In real-world, we dont have the meteorol data in the predicted timeslot, instead, we use the meteoral at previous timeslots, i.e., slot = predicted_slot - timeslot (you can use predicted meteorol data as well)
    '''
    fname = os.path.join(datapath, 'TaxiBJ', 'BJ_Meteorology.h5')
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

def load_data_TaxiBJ(T=48, nb_flow=2, len_closeness=None, len_period=None, len_trend=None,
              len_test=None, meta_data=True, meteorol_data=True,
              holiday_data=True, datapath=None):
    """
    """
    assert(len_closeness + len_period + len_trend > 0)
    # load data
    # 13 - 16
    data_all = []
    timestamps_all = list()
    for year in range(13, 17):
        fname = os.path.join(
            datapath, 'TaxiBJ', 'BJ{}_M32x32_T30_InOut.h5'.format(year))
        print("file name: ", fname)
        data, timestamps = load_stdata(fname)
        # print(timestamps)
        # remove a certain day which does not have 48 timestamps
        data, timestamps = remove_incomplete_days(data, timestamps, T)
        data = data[:, :nb_flow]
        data[data < 0] = 0.
        data_all.append(data)
        timestamps_all.append(timestamps)
        print("\n")

    # minmax_scale
    data_train = np.vstack(copy(data_all))[:-len_test]
    print('train_data shape: ', data_train.shape)
    mmn = MinMaxNormalization()
    mmn.fit(data_train)
    data_all_mmn = [mmn.transform(d) for d in data_all]

    XC, XP, XT = [], [], []
    Y = []
    timestamps_Y = []
    for data, timestamps in zip(data_all_mmn, timestamps_all):
        # instance-based dataset --> sequences with format as (X, Y) where X is
        # a sequence of images and Y is an image.
        st = STMatrix(data, timestamps, T, CheckComplete=False)
        _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset_3D(
            len_closeness=len_closeness, len_period=len_period, len_trend=len_trend)
        XC.append(_XC)
        XP.append(_XP)
        XT.append(_XT)
        Y.append(_Y)
        timestamps_Y += _timestamps_Y

    meta_feature = []
    if meta_data:
        # load time feature
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
    if metadata_dim < 1:
        metadata_dim = None
    if meta_data and holiday_data and meteorol_data:
        print('time feature:', time_feature.shape, 'holiday feature:', holiday_feature.shape,
              'meteorol feature: ', meteorol_feature.shape, 'mete feature: ', meta_feature.shape)

    XC = np.vstack(XC)
    XP = np.vstack(XP)
    XT = np.vstack(XT)
    Y = np.vstack(Y)
    print("XC shape: ", XC.shape, "XP shape: ", XP.shape,
          "XT shape: ", XT.shape, "Y shape:", Y.shape)

    XC_train, XP_train, XT_train, Y_train = XC[
        :-len_test], XP[:-len_test], XT[:-len_test], Y[:-len_test]
    XC_test, XP_test, XT_test, Y_test = XC[
        -len_test:], XP[-len_test:], XT[-len_test:], Y[-len_test:]
    timestamp_train, timestamp_test = timestamps_Y[
        :-len_test], timestamps_Y[-len_test:]

    X_train = []
    X_test = []
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_train, XP_train, XT_train]):
        if l > 0:
            X_train.append(X_)
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_test, XP_test, XT_test]):
        if l > 0:
            X_test.append(X_)
    print('train shape:', XC_train.shape, Y_train.shape,
          'test shape: ', XC_test.shape, Y_test.shape)

    if metadata_dim is not None:
        meta_feature_train, meta_feature_test = meta_feature[
            :-len_test], meta_feature[-len_test:]
        X_train.append(meta_feature_train)
        X_test.append(meta_feature_test)
    for _X in X_train:
        print(_X.shape, )
    print()
    for _X in X_test:
        print(_X.shape, )
    print()
    return X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test

def load_data_carRome(T=24*2, nb_flow=2, len_closeness=None, len_period=None, len_trend=None,
              len_test=None, preprocess_name='preprocessing.pkl',
              meta_data=True, meteorol_data=False, holiday_data=True, datapath=None, shape=(32,32)):
    assert(len_closeness + len_period + len_trend > 0)
    # load data
    if (shape == (32,32)):
        data, timestamps = load_stdata(os.path.join(datapath, 'Roma', 'AllMap', 'Roma_32x32_30_minuti_north.h5'))
    else:
        data, timestamps = load_stdata(os.path.join(datapath, 'Roma', 'AllMap', 'Roma_16x8_1_ora_resize_north.h5'))
    # print(len(timestamps))
    # remove a certain day which does not have 48 timestamps
    data, timestamps = remove_incomplete_days(data, timestamps, T)
    data = data[:, :nb_flow]
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

    XC, XP, XT = [], [], []
    Y = []
    timestamps_Y = []
    for data, timestamps in zip(data_all_mmn, timestamps_all):
        # instance-based dataset --> sequences with format as (X, Y) where X is a sequence of images and Y is an image.
        st = STMatrix(data, timestamps, T, CheckComplete=False)
        _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset_3D(len_closeness=len_closeness, len_period=len_period,
                                                                len_trend=len_trend)
        XC.append(_XC)
        XP.append(_XP)
        XT.append(_XT)
        Y.append(_Y)
        timestamps_Y += _timestamps_Y

    meta_feature = []
    if meta_data:
        # load time feature
        time_feature = timestamp2vec(timestamps_Y)
        meta_feature.append(time_feature)
    if holiday_data:
        # load holiday
        holiday_feature = load_holiday_rome(timestamps_Y, datapath)
        meta_feature.append(holiday_feature)
    if meteorol_data:
        # load meteorol data
        meteorol_feature = load_meteorol_rome(timestamps_Y, datapath)
        meta_feature.append(meteorol_feature)

    meta_feature = np.hstack(meta_feature) if len(
        meta_feature) > 0 else np.asarray(meta_feature)
    metadata_dim = meta_feature.shape[1] if len(
        meta_feature.shape) > 1 else None
    if metadata_dim < 1:
        metadata_dim = None
    if meta_data and holiday_data and meteorol_data:
        print('time feature:', time_feature.shape, 'holiday feature:', holiday_feature.shape,
              'meteorol feature: ', meteorol_feature.shape, 'mete feature: ', meta_feature.shape)

    XC = np.vstack(XC)
    XP = np.vstack(XP)
    XT = np.vstack(XT)
    Y = np.vstack(Y)
    print("XC shape: ", XC.shape, "XP shape: ", XP.shape,
          "XT shape: ", XT.shape, "Y shape:", Y.shape)

    XC_train, XP_train, XT_train, Y_train = XC[
        :-len_test], XP[:-len_test], XT[:-len_test], Y[:-len_test]
    XC_test, XP_test, XT_test, Y_test = XC[
        -len_test:], XP[-len_test:], XT[-len_test:], Y[-len_test:]
    timestamp_train, timestamp_test = timestamps_Y[
        :-len_test], timestamps_Y[-len_test:]

    X_train = []
    X_test = []
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_train, XP_train, XT_train]):
        if l > 0:
            X_train.append(X_)
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_test, XP_test, XT_test]):
        if l > 0:
            X_test.append(X_)
    print('train shape:', XC_train.shape, Y_train.shape,
          'test shape: ', XC_test.shape, Y_test.shape)

    if metadata_dim is not None:
        meta_feature_train, meta_feature_test = meta_feature[
            :-len_test], meta_feature[-len_test:]
        X_train.append(meta_feature_train)
        X_test.append(meta_feature_test)
    for _X in X_train:
        print(_X.shape, )
    print()
    for _X in X_test:
        print(_X.shape, )
    print()
    return X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test

def load_data_carRome2(T=24*2, nb_flow=2, len_closeness=None, len_period=None, len_trend=None,
              len_test=None, preprocess_name='preprocessing.pkl',
              meta_data=True, meteorol_data=False, holiday_data=True, datapath=None, shape=(32,32)):
    assert(len_closeness + len_period + len_trend > 0)
    # load data
    if (shape == (32, 32)):
        data, timestamps = load_stdata(os.path.join(datapath, 'Roma', 'Centro', 'Rome_32x32_30_minuti_centro.h5'))
    else:
        data, timestamps = load_stdata(os.path.join(datapath, 'Roma', 'Centro', 'Rome_16x8_1_ora_centro.h5'))
    # print(len(timestamps))
    # remove a certain day which does not have 48 timestamps
    data, timestamps = remove_incomplete_days(data, timestamps, T)
    data = data[:, :nb_flow]
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

    XC, XP, XT = [], [], []
    Y = []
    timestamps_Y = []
    for data, timestamps in zip(data_all_mmn, timestamps_all):
        # instance-based dataset --> sequences with format as (X, Y) where X is a sequence of images and Y is an image.
        st = STMatrix(data, timestamps, T, CheckComplete=False)
        _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset_3D(len_closeness=len_closeness, len_period=len_period,
                                                                len_trend=len_trend)
        XC.append(_XC)
        XP.append(_XP)
        XT.append(_XT)
        Y.append(_Y)
        timestamps_Y += _timestamps_Y

    meta_feature = []
    if meta_data:
        # load time feature
        time_feature = timestamp2vec(timestamps_Y)
        meta_feature.append(time_feature)
    if holiday_data:
        # load holiday
        holiday_feature = load_holiday_rome2(timestamps_Y, datapath)
        meta_feature.append(holiday_feature)
    if meteorol_data:
        # load meteorol data
        meteorol_feature = load_meteorol_rome2(timestamps_Y, datapath)
        meta_feature.append(meteorol_feature)

    meta_feature = np.hstack(meta_feature) if len(
        meta_feature) > 0 else np.asarray(meta_feature)
    metadata_dim = meta_feature.shape[1] if len(meta_feature.shape) > 1 else None
    if meta_data and holiday_data and meteorol_data:
        print('time feature:', time_feature.shape, 'holiday feature:', holiday_feature.shape,
              'meteorol feature: ', meteorol_feature.shape, 'mete feature: ', meta_feature.shape)

    XC = np.vstack(XC)
    XP = np.vstack(XP)
    XT = np.vstack(XT)
    Y = np.vstack(Y)
    print("XC shape: ", XC.shape, "XP shape: ", XP.shape,
          "XT shape: ", XT.shape, "Y shape:", Y.shape)

    XC_train, XP_train, XT_train, Y_train = XC[
        :-len_test], XP[:-len_test], XT[:-len_test], Y[:-len_test]
    XC_test, XP_test, XT_test, Y_test = XC[
        -len_test:], XP[-len_test:], XT[-len_test:], Y[-len_test:]
    timestamp_train, timestamp_test = timestamps_Y[
        :-len_test], timestamps_Y[-len_test:]

    X_train = []
    X_test = []
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_train, XP_train, XT_train]):
        if l > 0:
            X_train.append(X_)
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_test, XP_test, XT_test]):
        if l > 0:
            X_test.append(X_)
    print('train shape:', XC_train.shape, Y_train.shape,
          'test shape: ', XC_test.shape, Y_test.shape)

    if metadata_dim is not None:
        meta_feature_train, meta_feature_test = meta_feature[
            :-len_test], meta_feature[-len_test:]
        X_train.append(meta_feature_train)
        X_test.append(meta_feature_test)
    for _X in X_train:
        print(_X.shape, )
    print()
    for _X in X_test:
        print(_X.shape, )
    print()
    return X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test


def load_data_bikeNYC(filename, T=24, nb_flow=2, len_closeness=None, len_period=None, len_trend=None, len_test=None, meta_data=True, datapath=None):
    assert (len_closeness + len_period + len_trend > 0)
    # load data
    data, timestamps = load_stdata(os.path.join(datapath,'BikeNYC',filename))
    # remove a certain day which does not have 48 timestamps
    data, timestamps = remove_incomplete_days(data, timestamps, T)
    data = data[:, :nb_flow]
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

    XC, XP, XT = [], [], []
    Y = []
    timestamps_Y = []
    for data, timestamps in zip(data_all_mmn, timestamps_all):
        # instance-based dataset --> sequences with format as (X, Y) where X is a sequence of images and Y is an image.
        st = STMatrix(data, timestamps, T, CheckComplete=False)
        _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset_3D(len_closeness=len_closeness, len_period=len_period,
                                                                len_trend=len_trend)
        XC.append(_XC)
        XP.append(_XP)
        XT.append(_XT)
        Y.append(_Y)
        timestamps_Y += _timestamps_Y

    XC = np.vstack(XC)
    XP = np.vstack(XP)
    XT = np.vstack(XT)
    Y = np.vstack(Y)
    print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
    XC_train, XP_train, XT_train, Y_train = XC[:-len_test], XP[:-len_test], XT[:-len_test], Y[:-len_test]
    XC_test, XP_test, XT_test, Y_test = XC[-len_test:], XP[-len_test:], XT[-len_test:], Y[-len_test:]

    timestamp_train, timestamp_test = timestamps_Y[:-len_test], timestamps_Y[-len_test:]
    X_train = []
    X_test = []
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_train, XP_train, XT_train]):
        if l > 0:
            X_train.append(X_)
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_test, XP_test, XT_test]):
        if l > 0:
            X_test.append(X_)
    print('train shape:', XC_train.shape, Y_train.shape, 'test shape: ', XC_test.shape, Y_test.shape)
    # load meta feature
    if meta_data:
        meta_feature = timestamp2vec(timestamps_Y)
        metadata_dim = meta_feature.shape[1]
        meta_feature_train, meta_feature_test = meta_feature[:-len_test], meta_feature[-len_test:]
        X_train.append(meta_feature_train)
        X_test.append(meta_feature_test)
    else:
        metadata_dim = None
    for _X in X_train:
        print(_X.shape, )
    print()
    for _X in X_test:
        print(_X.shape, )
    print()
    return X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test

def load_data_taxiNYC(T=24, nb_flow=2, len_closeness=None, len_period=None, len_trend=None,
              len_test=None, meta_data=True, meteorol_data=False, holiday_data=False, datapath=None):
    """
    """
    assert(len_closeness + len_period + len_trend > 0)
    # load data
    # 10 - 14
    data_all = []
    timestamps_all = list()
    for year in range(10, 15):
        fname = os.path.join(
            datapath, 'TaxiNYC', 'NYC{}_Taxi_M16x8_T60_InOut.h5'.format(year))
        print("file name: ", fname)
        data, timestamps = load_stdata(fname)
        # print(timestamps)
        # remove a certain day which does not have 48 timestamps
        data, timestamps = remove_incomplete_days_taxiNYC(data, timestamps, T)
        data = data[:, :nb_flow]
        data[data < 0] = 0.
        data_all.append(data)
        timestamps_all.append(timestamps)
        print("\n")
    
    # minmax_scale
    data_train = np.vstack(copy(data_all))[:-len_test]
    print('train_data shape: ', data_train.shape)
    mmn = MinMaxNormalization()
    mmn.fit(data_train)
    data_all_mmn = [mmn.transform(d) for d in data_all]

    XC, XP, XT = [], [], []
    Y = []
    timestamps_Y = []
    for data, timestamps in zip(data_all_mmn, timestamps_all):
        # instance-based dataset --> sequences with format as (X, Y) where X is a sequence of images and Y is an image.
        st = STMatrix(data, timestamps, T, CheckComplete=False, Hours0_23=True)
        _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset_3D(len_closeness=len_closeness, len_period=len_period, len_trend=len_trend)
        XC.append(_XC)
        XP.append(_XP)
        XT.append(_XT)
        Y.append(_Y)
        timestamps_Y += _timestamps_Y

    XC = np.vstack(XC)
    XP = np.vstack(XP)
    XT = np.vstack(XT)
    Y = np.vstack(Y)
    print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
    XC_train, XP_train, XT_train, Y_train = XC[:-len_test], XP[:-len_test], XT[:-len_test], Y[:-len_test]
    XC_test, XP_test, XT_test, Y_test = XC[-len_test:], XP[-len_test:], XT[-len_test:], Y[-len_test:]
    
    timestamp_train, timestamp_test = timestamps_Y[:-len_test], timestamps_Y[-len_test:]
    X_train = []
    X_test = []
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_train, XP_train, XT_train]):
        if l > 0:
            X_train.append(X_)
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_test, XP_test, XT_test]):
        if l > 0:
            X_test.append(X_)
    print('train shape:', XC_train.shape, Y_train.shape, 'test shape: ', XC_test.shape, Y_test.shape)
    # load meta feature
    meta_feature = []
    if meta_data:
        time_feature = timestamp2vec(timestamps_Y)
        meta_feature.append(time_feature)
        if holiday_data:
            # load holiday
            holiday_feature = load_holiday_taxiNYC(timestamps_Y, datapath)
            meta_feature.append(holiday_feature)
        if meteorol_data:
            # load meteorol data
            meteorol_feature = load_meteorol_taxiNYC(timestamps_Y, datapath)
            meta_feature.append(meteorol_feature)

        meta_feature = np.hstack(meta_feature) if len(
            meta_feature) > 0 else np.asarray(meta_feature)
        metadata_dim = meta_feature.shape[1] if len(
            meta_feature.shape) > 1 else None
        metadata_dim = meta_feature.shape[1]
        meta_feature_train, meta_feature_test = meta_feature[:-len_test], meta_feature[-len_test:]
        X_train.append(meta_feature_train)
        X_test.append(meta_feature_test)
    else:
        metadata_dim = None
    for _X in X_train:
        print(_X.shape, )
    print()
    for _X in X_test:
        print(_X.shape, )
    print()
    return X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test


### load and cache bikeNYC data
# DATAPATH = '../data'
# T = 24  # number of time intervals in one day
# len_closeness = 6  # length of closeness dependent sequence
# len_period = 0  # length of peroid dependent sequence
# len_trend = 4  # length of trend dependent sequence
# nb_residual_unit = 4   # number of residual units
# nb_flow = 2  # there are two types of flows: new-flow and end-flow
# days_test = 10 # divide data into two subsets: Train & Test, of which the test set is the last 10 days
# len_test = T * days_test
# map_height, map_width = 16, 8  # grid size
# original_filename = 'NYC14_M16x8_T60_NewEnd.h5'

# X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = \
#         load_data_bikeNYC(original_filename, T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period,
#                   len_trend=len_trend, len_test=len_test, meta_data=False, datapath=DATAPATH)

# CACHEDATA=True
# path_cache = os.path.join(DATAPATH, 'CACHE', 'ST3DNet')
# if CACHEDATA and os.path.isdir(path_cache) is False:
#     os.mkdir(path_cache)
# filename = os.path.join(path_cache, 'BikeNYC_c%d_p%d_t%d_noext'%(len_closeness, len_period, len_trend))

# f = open(filename, 'wb')
# pickle.dump(X_train, f)
# pickle.dump(Y_train, f)
# pickle.dump(X_test, f)
# pickle.dump(Y_test, f)
# pickle.dump(mmn, f)
# pickle.dump(external_dim, f)
# pickle.dump(timestamp_train, f)
# pickle.dump(timestamp_test, f)
# f.close()
# ###

# ### load and cache TaxiBJ data
# DATAPATH = '../data'
# T = 48  # number of time intervals in one day
# len_closeness = 6  # length of closeness dependent sequence
# len_period = 0  # length of peroid dependent sequence
# len_trend = 2  # length of trend dependent sequence
# nb_residual_unit = 7   # number of residual units
# nb_flow = 2  # there are two types of flows: new-flow and end-flow
# days_test = 7*4
# len_test = T * days_test
# map_height, map_width = 32, 32  # grid size

# X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = \
#         load_data_TaxiBJ(T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period,
#                   len_trend=len_trend, len_test=len_test, meta_data=False, datapath=DATAPATH)

# CACHEDATA=True
# path_cache = os.path.join(DATAPATH, 'CACHE', 'ST3DNet')
# if CACHEDATA and os.path.isdir(path_cache) is False:
#     os.mkdir(path_cache)
# filename = os.path.join(path_cache, 'TaxiBJ_c%d_p%d_t%d_noext'%(len_closeness, len_period, len_trend))

# f = open(filename, 'wb')
# pickle.dump(X_train, f)
# pickle.dump(Y_train, f)
# pickle.dump(X_test, f)
# pickle.dump(Y_test, f)
# pickle.dump(mmn, f)
# pickle.dump(external_dim, f)
# pickle.dump(timestamp_train, f)
# pickle.dump(timestamp_test, f)
# f.close()
###

### load and cache taxiNYC data
# DATAPATH = '../data'
# T = 24  # number of time intervals in one day
# len_closeness = 6  # length of closeness dependent sequence
# len_period = 0  # length of peroid dependent sequence
# len_trend = 2  # length of trend dependent sequence
# nb_residual_unit = 4   # number of residual units
# nb_flow = 2  # there are two types of flows: new-flow and end-flow
# days_test = 7*4
# len_test = T * days_test
# map_height, map_width = 16, 8  # grid size

# X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = \
#         load_data_taxiNYC(T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period,
#                   len_trend=len_trend, len_test=len_test, meta_data=True, meteorol_data=True, holiday_data=True, datapath=DATAPATH)

# CACHEDATA=True
# path_cache = os.path.join(DATAPATH, 'CACHE', 'ST3DNet')
# if CACHEDATA and os.path.isdir(path_cache) is False:
#     os.mkdir(path_cache)
# filename = os.path.join(path_cache, 'TaxiNYC_c%d_p%d_t%d_noext'%(len_closeness, len_period, len_trend))

# f = open(filename, 'wb')
# pickle.dump(X_train, f)
# pickle.dump(Y_train, f)
# pickle.dump(X_test, f)
# pickle.dump(Y_test, f)
# pickle.dump(mmn, f)
# pickle.dump(external_dim, f)
# pickle.dump(timestamp_train, f)
# pickle.dump(timestamp_test, f)
# f.close()
###

### load and cache RomaNord32x32 data
DATAPATH = '../data'
T = 24*2  # number of time intervals in one day
len_closeness = 6  # length of closeness dependent sequence
len_period = 0  # length of peroid dependent sequence
len_trend = 2  # length of trend dependent sequence
nb_residual_unit = 7   # number of residual units
nb_flow = 2  # there are two types of flows: new-flow and end-flow
days_test = 7
len_test = T * days_test
map_height, map_width = 32, 32  # grid size

X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = \
        load_data_carRome(T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period,
                  len_trend=len_trend, len_test=len_test, meta_data=False, meteorol_data=False, holiday_data=True, datapath=DATAPATH)

CACHEDATA=True
path_cache = os.path.join(DATAPATH, 'CACHE', 'ST3DNet')
if CACHEDATA and os.path.isdir(path_cache) is False:
    os.mkdir(path_cache)
filename = os.path.join(path_cache, 'Rome_c%d_p%d_t%d_noext'%(len_closeness, len_period, len_trend))

f = open(filename, 'wb')
pickle.dump(X_train, f)
pickle.dump(Y_train, f)
pickle.dump(X_test, f)
pickle.dump(Y_test, f)
pickle.dump(mmn, f)
pickle.dump(external_dim, f)
pickle.dump(timestamp_train, f)
pickle.dump(timestamp_test, f)
f.close()
###

### load and cache RomaNord16x8 data
DATAPATH = '../data'
T = 24  # number of time intervals in one day
len_closeness = 6  # length of closeness dependent sequence
len_period = 0  # length of peroid dependent sequence
len_trend = 2  # length of trend dependent sequence
nb_residual_unit = 7   # number of residual units
nb_flow = 2  # there are two types of flows: new-flow and end-flow
days_test = 7
len_test = T * days_test
map_height, map_width = 16, 8  # grid size

X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = \
        load_data_carRome(T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period,
                  len_trend=len_trend, len_test=len_test, meta_data=True, meteorol_data=True, holiday_data=True, datapath=DATAPATH, shape=(16,8))

CACHEDATA=True
path_cache = os.path.join(DATAPATH, 'CACHE', 'ST3DNet')
if CACHEDATA and os.path.isdir(path_cache) is False:
    os.mkdir(path_cache)
filename = os.path.join(path_cache, 'Rome16x8_c%d_p%d_t%d_noext'%(len_closeness, len_period, len_trend))

f = open(filename, 'wb')
pickle.dump(X_train, f)
pickle.dump(Y_train, f)
pickle.dump(X_test, f)
pickle.dump(Y_test, f)
pickle.dump(mmn, f)
pickle.dump(external_dim, f)
pickle.dump(timestamp_train, f)
pickle.dump(timestamp_test, f)
f.close()
##

### load and cache Roma_Bergamo32x32 data
#DATAPATH = '../data'
#T = 24*2  # number of time intervals in one day
#len_closeness = 6  # length of closeness dependent sequence
#len_period = 0  # length of peroid dependent sequence
#len_trend = 2  # length of trend dependent sequence
#nb_residual_unit = 7   # number of residual units
#nb_flow = 2  # there are two types of flows: new-flow and end-flow
#days_test = 7
#len_test = T * days_test
#map_height, map_width = 32, 32  # grid size
#
#X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = \
#        load_data_carRome2(T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period,
#                  len_trend=len_trend, len_test=len_test, meta_data=False, meteorol_data=False, holiday_data=True, datapath=DATAPATH)
#
#CACHEDATA=True
#path_cache = os.path.join(DATAPATH, 'CACHE', 'ST3DNet')
#if CACHEDATA and os.path.isdir(path_cache) is False:
#    os.mkdir(path_cache)
#filename = os.path.join(path_cache, 'Rome_c%d_p%d_t%d_noext_2'%(len_closeness, len_period, len_trend))
#
#f = open(filename, 'wb')
#pickle.dump(X_train, f)
#pickle.dump(Y_train, f)
#pickle.dump(X_test, f)
#pickle.dump(Y_test, f)
#pickle.dump(mmn, f)
#pickle.dump(external_dim, f)
#pickle.dump(timestamp_train, f)
#pickle.dump(timestamp_test, f)
#f.close()
####
#
#### load and cache RomaNord16x8 data
#DATAPATH = '../data'
#T = 24  # number of time intervals in one day
#len_closeness = 6  # length of closeness dependent sequence
#len_period = 0  # length of peroid dependent sequence
#len_trend = 2  # length of trend dependent sequence
#nb_residual_unit = 7   # number of residual units
#nb_flow = 2  # there are two types of flows: new-flow and end-flow
#days_test = 7
#len_test = T * days_test
#map_height, map_width = 16, 8  # grid size
#
#X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = \
#        load_data_carRome2(T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period,
#                  len_trend=len_trend, len_test=len_test, meta_data=True, meteorol_data=True, holiday_data=True, datapath=DATAPATH, shape=(16,8))
#
#CACHEDATA=True
#path_cache = os.path.join(DATAPATH, 'CACHE', 'ST3DNet')
#if CACHEDATA and os.path.isdir(path_cache) is False:
#    os.mkdir(path_cache)
#filename = os.path.join(path_cache, 'Rome16x8_c%d_p%d_t%d_noext_2'%(len_closeness, len_period, len_trend))
#
#f = open(filename, 'wb')
#pickle.dump(X_train, f)
#pickle.dump(Y_train, f)
#pickle.dump(X_test, f)
#pickle.dump(Y_test, f)
#pickle.dump(mmn, f)
#pickle.dump(external_dim, f)
#pickle.dump(timestamp_train, f)
#pickle.dump(timestamp_test, f)
#f.close()
####