import h5py
import numpy as np
import math
from matplotlib import pyplot
import os
from copy import copy


def load_stdata(fname):
    f = h5py.File(fname, 'r')
    data = f['data'].value
    timestamps = f['date'].value
    f.close()
    return data, timestamps

def adjust_timeslots(timeslot):
    timeslot_str = timeslot.decode("utf-8")
    interval = timeslot_str[-2:]
    new_interval = f'{int(interval)+1:02}'
    return bytes(timeslot_str[:-2] + new_interval, encoding='utf8')
    # return bytes(timeslot_str[:-4] + timeslot_str[-2:], encoding='utf8')

def remove_incomplete_days(data, timestamps, T=48, h0_23=False):
    # remove a certain day which has not 48 timestamps
    days = []  # available days: some day only contain some seqs
    days_incomplete = []
    i = 0
    first_timestamp_index = 0 if h0_23 else 1
    last_timestamp_index = T-1 if h0_23 else T
    while i < len(timestamps):
        if int(timestamps[i][8:]) != first_timestamp_index:
            i += 1
        elif i+T-1 < len(timestamps) and int(timestamps[i+T-1][8:]) == last_timestamp_index:
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


def load_data_taxiNYC(datapath):
    nb_flow = 2 # i.e. inflow and outflow
    T = 24 # number timestamps per day

    # load data
    data_all = []
    timestamps_all = list()
    for year in range(10, 15):
        fname = os.path.join(
            datapath, 'TaxiNYC', 'NYC{}_Taxi_M16x8_T60_InOut.h5'.format(year))
        print("file name: ", fname)
        data, timestamps = load_stdata(fname)
        # print(timestamps)
        # remove a certain day which does not have 48 timestamps
        data, timestamps = remove_incomplete_days(data, timestamps, T, True)
        data = data[:, :nb_flow]
        data[data < 0] = 0.
        data_all.append(data)
        timestamps_all.append(timestamps)
    timestamps_all = [timestamp for l in timestamps_all for timestamp in l]
    data_all = np.vstack(copy(data_all))
    print('data shape: ' + str(data_all.shape))
    
    return data_all, timestamps_all


def load_data_bikeNYC(datapath):
    nb_flow = 2 # i.e. inflow and outflow
    T = 24 # number timestamps per day

    # load data
    fname = os.path.join(datapath, 'BikeNYC', 'NYC14_M16x8_T60_NewEnd.h5')
    print("file name: ", fname)
    data, timestamps = load_stdata(fname)
    # print(timestamps)
    # remove a certain day which does not have 48 timestamps
    data, timestamps = remove_incomplete_days(data, timestamps, T)
    data = data[:, :nb_flow]
    data[data < 0] = 0.
    print('data shape: ' + str(data.shape))

    return data, timestamps


def load_data_taxiBJ(datapath):
    nb_flow = 2 # i.e. inflow and outflow
    T = 48 # number timestamps per day

    # load data
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
    timestamps_all = [timestamp for l in timestamps_all for timestamp in l]
    data_all = np.vstack(copy(data_all))
    print('data shape: ' + str(data_all.shape))

    return data_all, timestamps_all

def load_data_Rome(datapath):
    nb_flow = 2 # i.e. inflow and outflow
    T = 24*4 # number timestamps per day

    # load data
    # fname = os.path.join(datapath, 'Roma', 'Centro', 'Roma_32x32_15_minuti.h5')
    fname = os.path.join(datapath, 'Roma', 'AllMap', 'Roma_32x32_15_minuti_north.h5')
    
    print("file name: ", fname)
    data, timestamps = load_stdata(fname)
    # timestamps = np.array([adjust_timeslots(t) for t in timestamps])

    # print(timestamps)
    # remove a certain day which does not have 48 timestamps
    data, timestamps = remove_incomplete_days(data, timestamps, T)
    data = data[:, :nb_flow]
    data[data < 0] = 0.
    print('data shape: ' + str(data.shape))

    return data, timestamps

def load_data_Rome_1ora(datapath):
    nb_flow = 2 # i.e. inflow and outflow
    T = 24 # number timestamps per day

    # load data
    fname = os.path.join(datapath, 'Roma', 'Roma_16x16_1_ora.h5')
    print("file name: ", fname)
    data, timestamps = load_stdata(fname)
    timestamps = np.array([adjust_timeslots(t) for t in timestamps])

    # print(timestamps)
    # remove a certain day which does not have 48 timestamps
    data, timestamps = remove_incomplete_days(data, timestamps, T)
    data = data[:, :nb_flow]
    data[data < 0] = 0.
    print('data shape: ' + str(data.shape))

    return data, timestamps