import matplotlib.pyplot as plt
import matplotlib.colors
import time
import datetime
import numpy as np
from keras import backend as K
from tensorflow.keras.metrics import (
    MeanAbsolutePercentageError
)

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

def plot_image(image, flow, cmap, vmax, dt=None):
    # image *= (255.0/image.max())
    im = plt.imshow(image[flow], cmap=cmap, interpolation='nearest', vmin=0, vmax=vmax)
    plt.colorbar(im)
    fl = 'Inflow' if flow==0 else 'Outflow'
    if (dt):
        plt.title(dt.strftime(f'%a %d %B %Y, %H:%M ({fl})'))
    # plt.show()
    # return im

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["green","yellow","red","#450903"])


def plot_measure_mean_by_region(real_data, predicted_data, flow, func):
    shape = real_data.shape
    measure_map = np.random.rand(shape[1], shape[2])
    for row in range(shape[1]):
        for col in range(shape[2]):
            measure_map[row, col] = func(real_data[:,row,col,flow], predicted_data[:,row,col,flow])

    measure_map[np.isnan(measure_map)] = 0
    vmax = measure_map.max()
    im = plt.imshow(measure_map, cmap='OrRd', interpolation='nearest', vmin=0, vmax=vmax)
    plt.colorbar(im)
    fl = 'Inflow' if flow==0 else 'Outflow'
    plt.title(f'{func.__name__.upper()} by region, {fl}')
    return measure_map


def get_index_of_date(timestamps, T, dt, first01=True):
    '''
    dt: datetime object
    '''

    # check if minute is valid considering T
    valid_minutes = [m for m in range(0, 60, int(60/(T/24)))]
    minute = dt.minute
    assert minute in valid_minutes, f'minute={minute} is not valid with T={T}'

    # build timestamp
    t_per_hour = len(valid_minutes)
    timeslot = dt.hour*t_per_hour + valid_minutes.index(minute)
    if (first01):
        timeslot += 1
    t = dt.strftime('%Y%m%d') + f'{timeslot:02}'
    t = bytes(t, 'utf8')

    # get index
    try:
        return timestamps.index(t) - 1
    except:
        raise Exception(f'{t} not in timestamps')

def timestamp_to_string(timestamp, format):
    t = timestamp.decode('utf8')
    dt = datetime.datetime.strptime(t[:8], '%Y%m%d')
    return dt.strftime(format)
