import numpy as np
import time
from keras import backend as K
import os
import h5py
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from src.carRome import load_data
from src.streednet import build_model
from utils import cache, read_cache
from src.evaluation import evaluate


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:  # Currently, memory growth needs to be the same across GPUs
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)  # Memory growth must be set before GPUs have been initialized

### 32x32
# params
DATAPATH = '../data'
CACHEDATA = True
T = 24*2  # number of time intervals in one day

len_closeness = 4  # length of closeness dependent sequence
len_period = 0  # length of peroid dependent sequence
len_trend = 0  # length of trend dependent sequence

nb_flow = 2  # there are two types of flows: in-flow and out-flow
days_test = 7
len_test = T * days_test
len_val = len_test # no validation

nb_epoch = 50
batch_size = 16

map_height, map_width = 32, 32  # grid size


# load data
cache_folder = 'Autoencoder/model3'
path_cache = os.path.join(DATAPATH, 'CACHE', cache_folder)  # cache path
path_model = 'MODEL'
if CACHEDATA and os.path.isdir(path_cache) is False:
    os.mkdir(path_cache)
if os.path.isdir('results') is False:
    os.mkdir('results')
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)


# load data
print("loading data...")
preprocess_name = 'preprocess_rome.pkl'
fname = os.path.join(path_cache, 'Rome_C{}_P{}_T{}.h5'.format(
    len_closeness, len_period, len_trend))
if os.path.exists(fname) and CACHEDATA:
    X_train_all, Y_train_all, X_train, Y_train, \
    X_val, Y_val, X_test, Y_test, mmn, external_dim, \
    timestamp_train_all, timestamp_train, timestamp_val, timestamp_test = read_cache(
        fname, preprocess_name)
    print("load %s successfully" % fname)
else:
    X_train_all, Y_train_all, X_train, Y_train, \
    X_val, Y_val, X_test, Y_test, mmn, external_dim, \
    timestamp_train_all, timestamp_train, timestamp_val, timestamp_test = load_data(
        T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,
        len_val=len_val, preprocess_name=preprocess_name, meta_data=True, datapath=DATAPATH)
    if CACHEDATA:
        cache(fname, X_train_all, Y_train_all, X_train, Y_train, X_val, Y_val, X_test, Y_test,
              external_dim, timestamp_train_all, timestamp_train, timestamp_val, timestamp_test)

print(external_dim)
print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])

def save_map(Y_pred, i, freeze= True, spatial = False):
    path_save = f'confronto_{map_height}x{map_width}'
    if os.path.isdir(path_save) is False:
        os.mkdir(path_save)
    # save real vs predicted
    if freeze:
        if not spatial:
            fname = f'Roma_{map_height}x{map_width}_trained_attention_iteration{i}.h5'
        else:
            fname = f'Roma_{map_height}x{map_width}_trained_only_spatial_iteration{i}.h5'
    else:
        fname = f'Roma_{map_height}x{map_width}_trained_random_weight_iteration{i}.h5'
    h5 = h5py.File(os.path.join(path_save,fname), 'w')
    h5.create_dataset('Y_real', data=Y_test)
    h5.create_dataset('Y_pred', data=Y_pred)
    h5.create_dataset('timestamps', data=timestamp_test)
    h5.create_dataset('max', data=mmn._max)
    h5.close()

def train_model(batch_size, encoder_block, filters, save_results=False, i='', freeze = True, spatial = False):
    # build model
    model = build_model(
        len_closeness, len_period, len_trend, nb_flow, map_height, map_width,
        external_dim=external_dim,
        encoder_blocks=encoder_block,
        filters= filters,
        kernel_size=3,
        num_res=2
    )
    if encoder_block==3:
        if freeze:
            #load weight
            model_fname = 'model3resunit_doppia_attention.TaxiBJ1.c4.p0.t0.encoderblocks_3.kernel_size_3.lr_0.0007.batchsize_16.noMeteo.best.h5'
            model.load_weights(os.path.join('../best_models', 'model3', model_fname))
            if not spatial:
                #freeze all layers except attention
                for layer in model.layers[:-28]:
                    layer.trainable = False
                hyperparams_name = 'Roma_32x32_iterazione{}_trained_attention_accuracy'.format(i)
            else:
                #freeze all layers except attention
                for layer in model.layers[:-13]:
                    layer.trainable = False
                hyperparams_name = 'Roma_32x32_iterazione{}_trained_only_spatial_accuracy'.format(i)
        else:
            hyperparams_name = 'Roma_32x32_iterazione{}_trained_random_weight_accuracy'.format(i)
    else:
        if freeze:
            # load weight
            model_fname = 'model3resunit_doppia_attention.TaxiNYC5.c4.p0.t0.encoderblocks_2.kernel_size_3.lr_0.00086.batchsize_48.best.h5'
            model.load_weights(os.path.join('../best_models', 'model3', model_fname))
            if not spatial:
                # freeze all layers except attention
                for layer in model.layers[:-28]:
                    layer.trainable = False
                hyperparams_name = 'Roma_16x8_iterazione{}_trained_attention_accuracy'.format(i)
            else:
                # freeze all layers except attention
                for layer in model.layers[:-13]:
                    layer.trainable = False
                hyperparams_name = 'Roma_16x8_iterazione{}_trained_only_attention_accuracy'.format(i)
        else:
            hyperparams_name = 'Roma_16x8_iterazione{}_trained_random_weight_accuracy'.format(i)


    fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))
    model_checkpoint = ModelCheckpoint(fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')

    # train model
    print("training model...")
    ts = time.time()
    print(f'Iteration {i}')
    np.random.seed(i * 18)
    tf.random.set_seed(i * 18)
    history = model.fit(X_train_all, Y_train_all,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        validation_data=(X_test, Y_test),
                        callbacks=[model_checkpoint],
                        verbose=0)
    print("\nelapsed time (training): %.3f seconds\n" % (time.time() - ts))
    tempo = time.time() - ts

    # evaluate
    model.load_weights(fname_param)
    score = model.evaluate(
        X_test, Y_test, batch_size=128, verbose=0)
    print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))

    if (save_results):
        print('evaluating using the model that has the best loss on the valid set')
        model.load_weights(fname_param)  # load best weights for current iteration

        Y_pred = model.predict(X_test)  # compute predictions
        score = evaluate(Y_test, Y_pred, mmn, rmse_factor=1)  # evaluate performance

        # save h5 file to generate map
        save_map(Y_pred, i, freeze, spatial)

        # save to csv
        if freeze:
            if not spatial:
                csv_name = os.path.join('results', f'Roma_{map_height}x{map_width}_trained_attention_results.csv')
            else:
                csv_name = os.path.join('results', f'Roma_{map_height}x{map_width}_trained_only_spatial_results.csv')
        else:
            csv_name = os.path.join('results', f'Roma_{map_height}x{map_width}_trained_random_weight_results.csv')
        if not os.path.isfile(csv_name):
            if os.path.isdir('results') is False:
                os.mkdir('results')
            with open(csv_name, 'a', encoding="utf-8") as file:
                file.write('iteration,'
                           'rsme_in,rsme_out,rsme_tot,'
                           'mape_in,mape_out,mape_tot,'
                           'ape_in,ape_out,ape_tot,'
                           'tempo_esecuzione'
                           )
                file.write("\n")
                file.close()
        with open(csv_name, 'a', encoding="utf-8") as file:
            file.write(f'{i},{score[0]},{score[1]},{score[2]},{score[3]},'
                       f'{score[4]},{score[5]},{score[6]},{score[7]},{score[8]},'
                       f'{tempo}'
                       )
            file.write("\n")
            file.close()
        K.clear_session()

# 10 iterations for training the two attention
for i in range(0, 10):
    train_model(batch_size=batch_size,
                encoder_block= 3,
                filters= [64, 64, 64, 64, 16],
                save_results=True,
                i=i)
# 10 iterations for training only spatial attention
for i in range(0, 10):
    train_model(batch_size=batch_size,
                encoder_block= 3,
                filters= [64, 64, 64, 64, 16],
                save_results=True,
                i=i,
                spatial=True)
# 10 iterations for training all the model with random weights
for i in range(0, 10):
    train_model(batch_size=batch_size,
                encoder_block=3,
                filters=[64, 64, 64, 64, 16],
                save_results=True,
                i=i,
                freeze=False)




### 16x8
T = 24
len_closeness = 4  # length of closeness dependent sequence
len_period = 0  # length of peroid dependent sequence
len_trend = 0  # length of trend dependent sequence
map_height, map_width = 16, 8  # grid size
days_test = 7
len_test = T * days_test
len_val = len_test # no validation

# load data
cache_folder = 'Autoencoder/model3'
path_cache = os.path.join(DATAPATH, 'CACHE', cache_folder)  # cache path
if CACHEDATA and os.path.isdir(path_cache) is False:
    os.mkdir(path_cache)

# load data
print("loading data...")
preprocess_name = 'preprocess_rome16x8.pkl'
fname = os.path.join(path_cache, 'Rome16x8_C{}_P{}_T{}.h5'.format(
    len_closeness, len_period, len_trend))
if os.path.exists(fname) and CACHEDATA:
    X_train_all, Y_train_all, X_train, Y_train, \
    X_val, Y_val, X_test, Y_test, mmn, external_dim, \
    timestamp_train_all, timestamp_train, timestamp_val, timestamp_test = read_cache(
        fname, preprocess_name)
    print("load %s successfully" % fname)
else:
    X_train_all, Y_train_all, X_train, Y_train, \
    X_val, Y_val, X_test, Y_test, mmn, external_dim, \
    timestamp_train_all, timestamp_train, timestamp_val, timestamp_test = load_data(
        T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,
        len_val=len_val, preprocess_name=preprocess_name, meta_data=True, meteorol_data=True, holiday_data=True, datapath=DATAPATH, shape=(16,8))
    if CACHEDATA:
        cache(fname, X_train_all, Y_train_all, X_train, Y_train, X_val, Y_val, X_test, Y_test,
              external_dim, timestamp_train_all, timestamp_train, timestamp_val, timestamp_test)

print(external_dim)
print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])

# build model

for i in range(0, 10):
    train_model(batch_size=batch_size,
                encoder_block= 2,
                filters= [64, 64, 64, 16],
                save_results=True,
                i=i)
# 10 iterations for training only spatial attention
for i in range(0, 10):
    train_model(batch_size=batch_size,
                encoder_block= 2,
                filters= [64, 64, 64, 16],
                save_results=True,
                i=i,
                spatial=True)
# 10 iterations for training all the model with random weights
for i in range(0, 10):
    train_model(batch_size=batch_size,
                encoder_block=2,
                filters=[64, 64, 64, 16],
                save_results=True,
                i=i,
                freeze=False)
