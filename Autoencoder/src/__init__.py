from __future__ import print_function
import h5py
import time
import pandas as pd
import numpy as np
from copy import copy
from datetime import datetime, timedelta

def load_stdata(fname):
    f = h5py.File(fname, 'r')
    data = f['data'].value
    timestamps = f['date'].value
    f.close()
    return data, timestamps


def stat(fname):
    def get_nb_timeslot(f):
        s = f['date'][0]
        e = f['date'][-1]
        year, month, day = map(int, [s[:4], s[4:6], s[6:8]])
        ts = time.strptime("%04i-%02i-%02i" % (year, month, day), "%Y-%m-%d")
        year, month, day = map(int, [e[:4], e[4:6], e[6:8]])
        te = time.strptime("%04i-%02i-%02i" % (year, month, day), "%Y-%m-%d")
        nb_timeslot = (time.mktime(te) - time.mktime(ts)) / (0.5 * 3600) + 48
        ts_str, te_str = time.strftime("%Y-%m-%d", ts), time.strftime("%Y-%m-%d", te)
        return nb_timeslot, ts_str, te_str

    with h5py.File(fname) as f:
        nb_timeslot, ts_str, te_str = get_nb_timeslot(f)
        nb_day = int(nb_timeslot / 48)
        mmax = f['data'].value.max()
        mmin = f['data'].value.min()
        stat = '=' * 5 + 'stat' + '=' * 5 + '\n' + \
               'data shape: %s\n' % str(f['data'].shape) + \
               '# of days: %i, from %s to %s\n' % (nb_day, ts_str, te_str) + \
               '# of timeslots: %i\n' % int(nb_timeslot) + \
               '# of timeslots (available): %i\n' % f['date'].shape[0] + \
               'missing ratio of timeslots: %.1f%%\n' % ((1. - float(f['date'].shape[0] / nb_timeslot)) * 100) + \
               'max: %.3f, min: %.3f\n' % (mmax, mmin) + \
               '=' * 5 + 'stat' + '=' * 5
        print(stat)

def string2timestamp(strings, T=48):
    timestamps = []

    time_per_slot = 24.0 / T
    num_per_T = T // 24
    for t in strings:
        year, month, day, slot = int(t[:4]), int(t[4:6]), int(t[6:8]), int(t[8:])-1
        timestamps.append(pd.Timestamp(datetime(year, month, day, hour=int(slot * time_per_slot), minute=(slot % num_per_T) * int(60.0 * time_per_slot))))

    return timestamps

def string2timestamp(strings, T=48):
    timestamps = []

    time_per_slot = 24.0 / T
    num_per_T = T // 24
    for t in strings:
        year, month, day, slot = int(t[:4]), int(t[4:6]), int(t[6:8]), int(t[8:])-1
        timestamps.append(pd.Timestamp(datetime(year, month, day, hour=int(slot * time_per_slot), minute=(slot % num_per_T) * int(60.0 * time_per_slot))))

    return timestamps

def timestamp2vec(timestamps):
    # tm_wday range [0, 6], Monday is 0
    # vec = [time.strptime(str(t[:8], encoding='utf-8'), '%Y%m%d').tm_wday for t in timestamps]  # python3
    vec = [time.strptime(t[:8].decode('ascii'), '%Y%m%d').tm_wday for t in timestamps]  # python2
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


def split_by_time(data, timestamps, split_timestamp):
    # divide data into two subsets:
    # e.g., Train: ~ 2015.06.21 & Test: 2015.06.22 ~ 2015.06.28
    assert(len(data) == len(timestamps))
    assert(split_timestamp in set(timestamps))

    data_1 = []
    timestamps_1 = []
    data_2 = []
    timestamps_2 = []
    switch = False
    for t, d in zip(timestamps, data):
        if split_timestamp == t:
            switch = True
        if switch is False:
            data_1.append(d)
            timestamps_1.append(t)
        else:
            data_2.append(d)
            timestamps_2.append(t)
    return (np.asarray(data_1), timestamps_1), (np.asarray(data_2), timestamps_2)


def timeseries2seqs(data, timestamps, length=3, T=48):
    raw_ts = copy(timestamps)
    if type(timestamps[0]) != pd.Timestamp:
        timestamps = string2timestamp(timestamps, T=T)

    offset = pd.DateOffset(minutes=24 * 60 // T)

    breakpoints = [0]
    for i in range(1, len(timestamps)):
        if timestamps[i-1] + offset != timestamps[i]:
            print(timestamps[i-1], timestamps[i], raw_ts[i-1], raw_ts[i])
            breakpoints.append(i)
    breakpoints.append(len(timestamps))
    X = []
    Y = []
    for b in range(1, len(breakpoints)):
        print('breakpoints: ', breakpoints[b-1], breakpoints[b])
        idx = range(breakpoints[b-1], breakpoints[b])
        for i in range(len(idx) - length):
            x = np.vstack(data[idx[i:i+length]])
            y = data[idx[i+length]]
            X.append(x)
            Y.append(y)
    X = np.asarray(X)
    Y = np.asarray(Y)
    print("X shape: ", X.shape, "Y shape:", Y.shape)
    return X, Y

def timeseries2seqs_meta(data, timestamps, length=3, T=48):
    raw_ts = copy(timestamps)
    if type(timestamps[0]) != pd.Timestamp:
        timestamps = string2timestamp(timestamps, T=T)

    offset = pd.DateOffset(minutes=24 * 60 // T)

    breakpoints = [0]
    for i in range(1, len(timestamps)):
        if timestamps[i-1] + offset != timestamps[i]:
            print(timestamps[i-1], timestamps[i], raw_ts[i-1], raw_ts[i])
            breakpoints.append(i)
    breakpoints.append(len(timestamps))
    X = []
    Y = []
    avail_timestamps = []
    for b in range(1, len(breakpoints)):
        print('breakpoints: ', breakpoints[b-1], breakpoints[b])
        idx = range(breakpoints[b-1], breakpoints[b])
        for i in range(len(idx) - length):
            avail_timestamps.append(raw_ts[idx[i+length]])
            x = np.vstack(data[idx[i:i+length]])
            y = data[idx[i+length]]
            X.append(x)
            Y.append(y)
    X = np.asarray(X)
    Y = np.asarray(Y)
    print("X shape: ", X.shape, "Y shape:", Y.shape)
    return X, Y, avail_timestamps


def timeseries2seqs_peroid_trend(data, timestamps, length=3, T=48, peroid=pd.DateOffset(days=7), peroid_len=2):
    raw_ts = copy(timestamps)
    if type(timestamps[0]) != pd.Timestamp:
        timestamps = string2timestamp(timestamps, T=T)

    # timestamps index
    timestamp_idx = dict()
    for i, t in enumerate(timestamps):
        timestamp_idx[t] = i

    offset = pd.DateOffset(minutes=24 * 60 // T)

    breakpoints = [0]
    for i in range(1, len(timestamps)):
        if timestamps[i-1] + offset != timestamps[i]:
            print(timestamps[i-1], timestamps[i], raw_ts[i-1], raw_ts[i])
            breakpoints.append(i)
    breakpoints.append(len(timestamps))
    X = []
    Y = []
    for b in range(1, len(breakpoints)):
        print('breakpoints: ', breakpoints[b-1], breakpoints[b])
        idx = range(breakpoints[b-1], breakpoints[b])
        for i in range(len(idx) - length):
            # period
            target_timestamp = timestamps[i+length]

            legal_idx = []
            for pi in range(1, 1+peroid_len):
                if target_timestamp - peroid * pi not in timestamp_idx:
                    break
                legal_idx.append(timestamp_idx[target_timestamp - peroid * pi])
            # print("len: ", len(legal_idx), peroid_len)
            if len(legal_idx) != peroid_len:
                continue

            legal_idx += idx[i:i+length]

            # trend
            x = np.vstack(data[legal_idx])
            y = data[idx[i+length]]
            X.append(x)
            Y.append(y)
    X = np.asarray(X)
    Y = np.asarray(Y)
    print("X shape: ", X.shape, "Y shape:", Y.shape)
    return X, Y


def timeseries2seqs_3D(data, timestamps, length=3, T=48):
    raw_ts = copy(timestamps)
    if type(timestamps[0]) != pd.Timestamp:
        timestamps = string2timestamp(timestamps, T=T)

    offset = pd.DateOffset(minutes=24 * 60 // T)

    breakpoints = [0]
    for i in range(1, len(timestamps)):
        if timestamps[i-1] + offset != timestamps[i]:
            print(timestamps[i-1], timestamps[i], raw_ts[i-1], raw_ts[i])
            breakpoints.append(i)
    breakpoints.append(len(timestamps))
    X = []
    Y = []
    for b in range(1, len(breakpoints)):
        print('breakpoints: ', breakpoints[b-1], breakpoints[b])
        idx = range(breakpoints[b-1], breakpoints[b])
        for i in range(len(idx) - length):
            x = data[idx[i:i+length]].reshape(-1, length, 32, 32)
            y = np.asarray([data[idx[i+length]]]).reshape(-1, 1, 32, 32)
            X.append(x)
            Y.append(y)
    X = np.asarray(X)
    Y = np.asarray(Y)
    print("X shape: ", X.shape, "Y shape:", Y.shape)
    return X, Y


def bug_timeseries2seqs(data, timestamps, length=3, T=48):
    # have a bug
    if type(timestamps[0]) != pd.Timestamp:
        timestamps = string2timestamp(timestamps, T=T)

    offset = pd.DateOffset(minutes=24 * 60 // T)

    breakpoints = [0]
    for i in range(1, len(timestamps)):
        if timestamps[i-1] + offset != timestamps[i]:
            breakpoints.append(i)
    X = []
    Y = []
    for b in range(1, len(breakpoints)):
        print('breakpoints: ', breakpoints[b-1], breakpoints[b])
        idx = range(breakpoints[b-1], breakpoints[b])
        for i in range(len(idx) - 3):
            x = np.vstack(data[idx[i:i+3]])
            y = data[idx[i+3]]
            X.append(x)
            Y.append(y)
    X = np.asarray(X)
    Y = np.asarray(Y)
    print("X shape: ", X.shape, "Y shape:", Y.shape)
    return X, Y