from copy import copy
import numpy as np
import os
import datetime

from utils import (
    load_stdata, remove_incomplete_days, evaluate, plot_region_data, save_to_csv
)

def get_day_of_week(timestamp):
    date_string = timestamp.decode("utf-8")[:-2]
    day_of_week = datetime.datetime.strptime(date_string, '%Y%m%d').strftime('%A')
    return day_of_week

def ha_prediction(data, timestamps, T, len_test):
    num_timestamps = len(data)

    # estraggo i dati di train. solo questi e le previsioni già effettuate
    # vengono usate per predire i nuovi valori
    train_data = list(data[:-len_test])

    predicted_data = []
    # loop su tutti i timestamp del test_set
    for i in range(num_timestamps-len_test, num_timestamps):
        # prendo tutti i timestamps corrispondenti alla stessa ora e allo stesso giorno
        # e faccio la media. Problema: ci sono dei giorni mancanti nel dataset
        # step = T * 7
        # start_idx = i % step
        # historical_data = [data_all[t] for t in range(start_idx, i, step)]

        # provo a usare semplicemente tutti i giorni precedenti alla stessa ora
        # step = T
        # start_idx = i % step
        # historical_data = [data_all[t] for t in range(start_idx, i, step)]
        # prediction = np.mean(historical_data, axis=0).astype(int)
        # predicted_data.append(prediction)

        # possibile soluzione: converto il corrispondete timestamp in giorno della
        # settimana e vedo se corrisponde
        # Ad esempio se T=24 e i=500, prendo i seguenti timestamp:
        # [20, 44, 68, 92, 116, 140, 164, 188, 212, 236, 260, 284, 308, 332, 356, 380, 404, 428, 452, 476]
        # e li considero solo se appartengono allo stesso giorno della settimana
        # del timestamp i 
        current_day_of_week = get_day_of_week(timestamps[i])
        step = T
        start_idx = i % step
        historical_data = [
            train_data[t] for t in range(start_idx, i, step) if get_day_of_week(timestamps[t]) == current_day_of_week
        ]
        prediction = np.mean(historical_data, axis=0).astype(int)
        train_data.append(prediction)
        predicted_data.append(prediction)

    predicted_data = np.asarray(predicted_data)
    print('prediction shape: ' + str(predicted_data.shape))
    return predicted_data

def ha_prediction_taxiBJ():
    DATAPATH = '../data'
    nb_flow = 2 # i.e. inflow and outflow
    T = 48 # number timestamps per day
    len_test = T * 4 * 7 # number of timestamps to predict (four weeks)

    # load data
    data_all = []
    timestamps_all = list()
    for year in range(13, 17):
        fname = os.path.join(
            DATAPATH, 'TaxiBJ', 'BJ{}_M32x32_T30_InOut.h5'.format(year))
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

    # make predictions
    predicted_data = ha_prediction(data_all, timestamps_all, T, len_test)

    # evaluate
    print('Evaluating on TaxiBJ')
    real_data = data_all[-len_test:]
    score = evaluate(real_data, predicted_data)

    # save to csv
    save_to_csv('HA', 'TaxiBJ', score)


def ha_prediction_bikeNYC():
    DATAPATH = '../data'
    nb_flow = 2 # i.e. inflow and outflow
    T = 24 # number timestamps per day
    len_test = T * 10 # number of timestamps to predict (ten days)

    # load data
    fname = os.path.join(DATAPATH, 'BikeNYC', 'NYC14_M16x8_T60_NewEnd.h5')
    print("file name: ", fname)
    data, timestamps = load_stdata(fname)
    # print(timestamps)
    # remove a certain day which does not have 48 timestamps
    data, timestamps = remove_incomplete_days(data, timestamps, T)
    data = data[:, :nb_flow]
    data[data < 0] = 0.
    print('data shape: ' + str(data.shape))

    # make predictions
    predicted_data = ha_prediction(data, timestamps, T, len_test)

    # evaluate
    print('Evaluating on BikeNYC')
    real_data = data[-len_test:]
    score = evaluate(real_data, predicted_data)

    # plot real vs prediction data of a region
    # plot_region_data(real_data, predicted_data, (13,3), 0)

    # save to csv
    save_to_csv('HA', 'BikeNYC', score)


def ha_prediction_taxiNYC():
    DATAPATH = '../data'
    nb_flow = 2 # i.e. inflow and outflow
    T = 24 # number timestamps per day
    len_test = T * 4 * 7 # number of timestamps to predict (four weeks)

    # load data
    data_all = []
    timestamps_all = list()
    for year in range(10, 15):
        fname = os.path.join(
            DATAPATH, 'TaxiNYC', 'NYC{}_Taxi_M16x8_T60_InOut.h5'.format(year))
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

    # make predictions
    predicted_data = ha_prediction(data_all, timestamps_all, T, len_test)

    # evaluate
    print('Evaluating on TaxiNYC')
    real_data = data_all[-len_test:]
    score = evaluate(real_data, predicted_data)

    ## save to csv
    save_to_csv('HA', 'TaxiNYC', score)


def ha_prediction_RomaNord():
    datapath = '../data'
    nb_flow = 2 # i.e. inflow and outflow
    T = 48 # number timestamps per day
    len_test = T * 7 # number of timestamps to predict

    # load data
    fname = os.path.join(datapath, 'Roma', 'AllMap', 'Roma_32x32_30_minuti_north.h5')
    
    print("file name: ", fname)
    data, timestamps = load_stdata(fname)
    # timestamps = np.array([adjust_timeslots(t) for t in timestamps])

    # print(timestamps)
    # remove a certain day which does not have 48 timestamps
    data, timestamps = remove_incomplete_days(data, timestamps, T)
    data = data[:, :nb_flow]
    data[data < 0] = 0.
    print('data shape: ' + str(data.shape))

    # make predictions
    predicted_data = ha_prediction(data, timestamps, T, len_test)

    # evaluate
    print('Evaluating on RomaNord')
    real_data = data[-len_test:]
    score = evaluate(real_data, predicted_data)

    # plot real vs prediction data of a region
    # plot_region_data(real_data, predicted_data, (13,3), 0)

    # save to csv
    save_to_csv('HA', 'RomaNord', score)

def ha_prediction_RomaNord16x8():
    datapath = '../data'
    nb_flow = 2 # i.e. inflow and outflow
    T = 24 # number timestamps per day
    len_test = T * 7 # number of timestamps to predict

    # load data
    fname = os.path.join(datapath, 'Roma', 'AllMap', 'Roma_16x8_1_ora_resize_north.h5')
    
    print("file name: ", fname)
    data, timestamps = load_stdata(fname)
    # timestamps = np.array([adjust_timeslots(t) for t in timestamps])

    # print(timestamps)
    # remove a certain day which does not have 48 timestamps
    data, timestamps = remove_incomplete_days(data, timestamps, T)
    data = data[:, :nb_flow]
    data[data < 0] = 0.
    print('data shape: ' + str(data.shape))

    # make predictions
    predicted_data = ha_prediction(data, timestamps, T, len_test)

    # evaluate
    print('Evaluating on RomaNord16x8')
    real_data = data[-len_test:]
    score = evaluate(real_data, predicted_data)

    # plot real vs prediction data of a region
    # plot_region_data(real_data, predicted_data, (13,3), 0)

    # save to csv
    save_to_csv('HA', 'RomaNord16x8', score)

if __name__ == '__main__':
    ha_prediction_taxiBJ()
    # ha_prediction_bikeNYC()
    ha_prediction_taxiNYC()
    # ha_prediction_RomaNord()
    # ha_prediction_RomaNord16x8()
