import h5py
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
import math
from matplotlib import pyplot
import os
from keras import backend as K
from tensorflow.keras.metrics import (
    MeanAbsolutePercentageError
)

def load_stdata(fname):
    f = h5py.File(fname, 'r')
    data = f['data'].value
    timestamps = f['date'].value
    f.close()
    return data, timestamps

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

def split_flow(X):
    inflow = X[:,0]
    outflow = X[:,1]
    return inflow, outflow

def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

def mape(y_true, y_pred):
    idx = y_true > 10

    m = MeanAbsolutePercentageError()
    m.update_state(y_true[idx], y_pred[idx])
    return m.result().numpy()
    # return np.mean(np.abs((y_true[idx] - y_pred[idx]) / y_true[idx])) * 100

def ape(y_true, y_pred):
    idx = y_true > 10
    return np.sum(np.abs((y_true[idx] - y_pred[idx]) / y_true[idx])) * 100

def mae(y_true, y_pred):
        """calculate mae loss"""
        return np.mean(np.abs(y_true - y_pred))
    
def evaluate(y_true, y_pred):

    y_true_in, y_true_out = split_flow(y_true)
    y_pred_in, y_pred_out = split_flow(y_pred)

    score = []

    score.append(rmse(y_true_in, y_pred_in))
    score.append(rmse(y_true_out, y_pred_out))
    score.append(rmse(y_true, y_pred))
    score.append(mape(y_true_in, y_pred_in))
    score.append(mape(y_true_out, y_pred_out))
    score.append(mape(y_true, y_pred))
    score.append(ape(y_true_in, y_pred_in))
    score.append(ape(y_true_out, y_pred_out))
    score.append(ape(y_true, y_pred))
    score.append(mae(y_true, y_pred))
    print(
        f'rmse_in: {score[0]}\n'
        f'rmse_out: {score[1]}\n'
        f'rmse_total: {score[2]}\n'
        f'mape_in: {score[3]}\n'
        f'mape_out: {score[4]}\n'
        f'mape_total: {score[5]}\n'
        f'ape_out: {score[6]}\n'
        f'ape_out: {score[7]}\n'
        f'ape_total: {score[8]}\n'
        f'mae_total: {score[9]}\n'
    )
    return score

def plot_region_data(real_data, predicted_data, region, flow):
    # region deve essere una lista o tupla di 2 elementi
    # flow deve essere 0 (inflow) o 1 (outflow)
    row, column = region[0], region[1]

    real_data_region = [x[flow][row][column] for x in real_data]
    predicted_data_region = [x[flow][row][column] for x in predicted_data]

    pyplot.plot(real_data_region)
    pyplot.plot(predicted_data_region, color='red')
    pyplot.legend(['real', 'predicted'])
    pyplot.show()

def save_to_csv(model_name, dataset_name, score):
    csv_name = f'{model_name}_results.csv'
    if not os.path.isfile(csv_name):
        with open(csv_name, 'a', encoding = "utf-8") as file:
            file.write(
                'dataset,'
                'rsme_in,rsme_out,rsme_tot,'
                'mape_in,mape_out,mape_tot,'
                'ape_in,ape_out,ape_tot'
                'mae_tot'
            )
            file.write("\n")
            file.close()
    with open(csv_name, 'a', encoding = "utf-8") as file:
        file.write(f'{dataset_name},{score[0]},{score[1]},{score[2]},{score[3]},'
                f'{score[4]},{score[5]},{score[6]},{score[7]},{score[8]},{score[9]}'
                )
        file.write("\n")
        file.close()