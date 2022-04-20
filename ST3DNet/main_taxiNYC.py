from ST3DNet import *
import pickle
from utils import *
import os
import json
import time
from bayes_opt import BayesianOptimization
import math
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import ParameterGrid

from evaluation import evaluate

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

nb_epoch = 100  # number of epoch at training stage
# batch_size = [16,32,64]  # batch size
T = 24  # number of time intervals in one day
# lr = [0.00015, 0.00035]  # learning rate
# lr = 0.00002  # learning rate
len_closeness = len_c =  6  # length of closeness dependent sequence
len_period = len_p = 0  # length of peroid dependent sequence
len_trend = len_t = 4  # length of trend dependent sequence
# nb_residual_unit = [4,5,6]   # number of residual units
nb_flow = 2  # there are two types of flows: new-flow and end-flow
days_test = 7*4  
len_test = T * days_test
map_height, map_width = 16, 8  # grid size

path_result = 'RET'
path_model = 'MODEL'
if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)

filename = os.path.join("../data", 'CACHE', 'ST3DNet', 'TaxiNYC_c%d_p%d_t%d_noext'%(len_closeness, len_period, len_trend))
f = open(filename, 'rb')
X_train = pickle.load(f)
Y_train = pickle.load(f)
X_test = pickle.load(f)
Y_test = pickle.load(f)
mmn = pickle.load(f)
external_dim = pickle.load(f)
timestamp_train = pickle.load(f)
timestamp_test = pickle.load(f)

for i in X_train:
    print(i.shape)

Y_train = mmn.inverse_transform(Y_train)  # X is MaxMinNormalized, Y is real value
Y_test = mmn.inverse_transform(Y_test)

c_conf = (len_closeness, nb_flow, map_height,
              map_width) if len_closeness > 0 else None
t_conf = (len_trend, nb_flow, map_height,
          map_width) if len_trend > 0 else None


def train_model(lr, batch_size, residual_units, save_results=False, i=''):
    # get discrete parameters
    residual_units = int(residual_units)
    batch_size = 16 * int(batch_size)
    # kernel_size = int(kernel_size)
    lr = round(lr,5)

    # build model
    model = ST3DNet(c_conf=c_conf, t_conf=t_conf, external_dim=external_dim,
                        nb_residual_unit=residual_units)
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[rmse])
    # model.summary()
    hyperparams_name = 'TaxiNYC{}.c{}.p{}.t{}.resunits_{}.lr_{}.batchsize_{}'.format(
        i, len_c, len_p, len_t, residual_units,
        lr, batch_size)
    fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))

    early_stopping = EarlyStopping(monitor='val_rmse', patience=25, mode='min')
    # lr_callback = LearningRateScheduler(lrschedule)
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')

    # train model
    print("training model...")
    ts = time.time()
    if (i):
        print(f'Iteration {i}')
        np.random.seed(i * 18)
        tf.random.set_seed(i * 18)
    history = model.fit(X_train, Y_train,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        validation_data=(X_test, Y_test),
                        # callbacks=[early_stopping, model_checkpoint],
                        # callbacks=[model_checkpoint, lr_callback],
                        callbacks=[model_checkpoint],
                        verbose=2)
    model.save_weights(os.path.join(
        'MODEL', '{}.h5'.format(hyperparams_name)), overwrite=True)
    pickle.dump((history.history), open(os.path.join(
        path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))
    print("\nelapsed time (training): %.3f seconds\n" % (time.time() - ts))

    # evaluate
    model.load_weights(fname_param)
    score = model.evaluate(
        X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))

    if (save_results):
        print('evaluating using the model that has the best loss on the valid set')
        model.load_weights(fname_param)  # load best weights for current iteration

        Y_pred = model.predict(X_test)  # compute predictions

        score = evaluate(Y_test, Y_pred, rmse_factor=1)  # evaluate performance

        # save to csv
        csv_name = os.path.join('results', 'st3dnet_taxiNYC_results.csv')
        if not os.path.isfile(csv_name):
            if os.path.isdir('results') is False:
                os.mkdir('results')
            with open(csv_name, 'a', encoding="utf-8") as file:
                file.write('iteration,'
                            'rsme_in,rsme_out,rsme_tot,'
                            'mape_in,mape_out,mape_tot,'
                            'ape_in,ape_out,ape_tot,mae_tot'
                            )
                file.write("\n")
                file.close()
        with open(csv_name, 'a', encoding="utf-8") as file:
            file.write(f'{i},{score[0]},{score[1]},{score[2]},{score[3]},'
                        f'{score[4]},{score[5]},{score[6]},{score[7]},{score[8]},{score[9]}'
                        )
            file.write("\n")
            file.close()
        K.clear_session()

    # bayes opt is a maximization algorithm, to minimize validation_loss, return 1-this
    bayes_opt_score = 1.0 - score[1]

    return bayes_opt_score

# bayesian optimization
optimizer = BayesianOptimization(f=train_model,
                                 pbounds={'residual_units': (4, 6.999),
                                          'lr': (0.0001,0.001),
                                          'batch_size': (1, 2.999), # *16
                                        #   'kernel_size': (3, 5.999)
                                 },
                                 verbose=2)

optimizer.maximize(init_points=2, n_iter=10)

# training-test-evaluation iterations with best params
if os.path.isdir('results') is False:
    os.mkdir('results')
targets = [e['target'] for e in optimizer.res]
bs_fname = 'bs_taxiNYC.json'
with open(os.path.join('results', bs_fname), 'w') as f:
    json.dump(optimizer.res, f, indent=2)
best_index = targets.index(max(targets))
params = optimizer.res[best_index]['params']
# save best params
params_fname = 'st3dnet_taxiNYC_best_params.json'
with open(os.path.join('results', params_fname), 'w') as f:
    json.dump(params, f, indent=2)
# with open(os.path.join('results', params_fname), 'r') as f:
#     params = json.load(f)
for i in range(0, 1):
    train_model(residual_units=params['residual_units'],
                lr=params['lr'],
                batch_size=params['batch_size'],
                # kernel_size=params['kernel_size'],
                save_results=True,
                i=i)
