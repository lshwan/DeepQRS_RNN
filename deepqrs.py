# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 21:54:06 2020

@author: LSH
"""

import tensorflow as tf
from tensorflow.keras import regularizers, activations
from tensorflow.keras.layers import SimpleRNN, Dense, TimeDistributed, LSTM, Bidirectional, Activation, Conv1D
from tensorflow.keras.models import Sequential, load_model
import tensorflow.keras.backend as K
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
import load_db

#tf.debugging.set_log_device_placement(True)

class deepqrs():
    data_set = []
    data_anno = []
    fs = 0
    deepqrs_model = None

    def __init__(self, pre_model=None):
        self.__database = ['ahadb', 'mitdb', 'edb']
        self.__load_db = load_db.load_db()
        self.deepqrs_model = pre_model

    def __set_db(self, db_list):
        self.__database = list(db_list)

    def load_data(self, dbs=[], fs=250, ad_gain=200):
        if len(dbs) > 0:
            self.__set_db(dbs)

        self.data_set = self.__load_db.load_full_db_data(self.__database)
        self.data_anno, _ = self.__load_db.load_full_db_anno(self.__database)
        self.fs = fs

    def pre_processing(self, data):
        b, a = sp.butter(2, [55.0 / (0.5 * self.fs), 65.0 / (0.5 * self.fs)], btype='bs')

        data = sp.lfilter(b, a, data, axis=0)

        b, a = sp.butter(2, [45.0 / (0.5 * self.fs), 55.0 / (0.5 * self.fs)], btype='bs')

        data = sp.lfilter(b, a, data, axis=0)

        b, a = sp.butter(2, 1.5 / (0.5 * self.fs), btype='high')

        data = sp.lfilter(b, a, data, axis=0)

# =============================================================================
#         b, a = sp.butter(2, 80.0 / (0.5 * self.fs), btype='low')
# =============================================================================
        b, a = sp.butter(2, 30.0 / (0.5 * self.fs), btype='low')

        data = sp.lfilter(b, a, data, axis=0)

        return data

    def make_data_sample(self, data, dim_input=5):
        data = self.pre_processing(data)
        col, row = data.shape
        ret_data = []

        for i in range(row):
            temp_data = np.ndarray((col - dim_input, dim_input))

            for j in range(dim_input):
                temp_data[:, j] = data[j:j-dim_input, i]

            ret_data.append(temp_data)

        return ret_data

    def get_rnn_data_set(self, dim_input=5, anno_range_ms=60, timestep=10000, direction='full', augment_on=True, sparse=False, flip=False):
        assert self.data_set, 'Load data first'

        ms_in_sample = (anno_range_ms * self.fs) // 1000
        timestep_in_sample = (timestep * self.fs) // 1000
        mit_rec_list = self.__load_db.get_record_list('mitdb')

        while True:
            for (rec, data), anno in zip(self.data_set.items(), self.data_anno.values()):
                anno_sample = np.array(anno['sample'])

                col, row = data.shape

                assert col > dim_input, "dim_input can't be greater than data length"

                data = self.pre_processing(data)

                if flip:
                    data = np.flipud(data)

                for i in range(row):
                    temp_data = np.ndarray((col - dim_input, dim_input))
                    temp_anno = np.zeros((col, ))

                    if direction == 'full':
                        for j in range(2*ms_in_sample):
                            if rec in mit_rec_list:
                                anno_idx = np.array(anno_sample) - ms_in_sample + j
                            else:
                                anno_idx = np.array(anno_sample) + j

                            anno_idx[anno_idx >= data.shape[0]] = -1

                            temp_anno[anno_idx[anno_idx >= 0]] = 1
                    elif direction == 'half':
                        for j in range(ms_in_sample):
                            if rec in mit_rec_list:
                                anno_idx = np.array(anno_sample) + j
                            else:
                                anno_idx = np.array(anno_sample) + ms_in_sample + j

                            anno_idx[anno_idx >= data.shape[0]] = -1

                            temp_anno[anno_idx[anno_idx >= 0]] = 1
                    elif direction == 'back_half':
                        for j in range(ms_in_sample):
                            if rec in mit_rec_list:
                                anno_idx = np.array(anno_sample) - j
                            else:
                                anno_idx = np.array(anno_sample) + ms_in_sample - j

                            anno_idx[anno_idx >= data.shape[0]] = -1

                            temp_anno[anno_idx[anno_idx >= 0]] = 1

                    if flip:
                        temp_anno = np.flipud(temp_anno)

                    for j in range(dim_input):
                        temp_data[:, j] = data[j:j-dim_input, i]

                    if augment_on:
                        aug_data = self.data_augmentation(temp_data)
                    else:
                        aug_data = [temp_data]

                    temp_size = temp_data.shape[0]

                    if sparse:
                        temp_anno = np.reshape(
                                    temp_anno[dim_input-1:-1][:timestep_in_sample * (temp_size // timestep_in_sample)],
                                (-1, timestep_in_sample, 1))
                    else:
                        temp_anno = np.reshape(
                                tf.keras.utils.to_categorical(
                                    temp_anno[dim_input-1:-1][:timestep_in_sample * (temp_size // timestep_in_sample)],
                                    num_classes=2),
                                (-1, timestep_in_sample, 2))

                    for temp_data in aug_data:
                        temp_data = np.reshape(
                                temp_data[:timestep_in_sample * (temp_size // timestep_in_sample), :],
                                (-1, timestep_in_sample, dim_input))

                        seg_num = np.cumprod(temp_data.shape)[-1] // (10*250*5*32)
                        if seg_num == 0:
                            iter_list = []
                        else:
                            iter_list = range(np.cumprod(temp_data.shape)[0] // seg_num - 1)

                        for i in iter_list:
                            yield (temp_data[i*seg_num:(i+1)*seg_num], temp_anno[i*seg_num:(i+1)*seg_num])
                        else:
                            yield (temp_data[i*seg_num:], temp_anno[i*seg_num:])

    def get_rnn_data_set_sampling(self, dim_input=5, anno_range_ms=60, timestep=10000, direction='full', augment_on=True, sparse=False, flip=False):
        assert self.data_set, 'Load data first'

        ms_in_sample = (anno_range_ms * self.fs) // 1000
        timestep_in_sample = (timestep * self.fs) // 1000
        mit_rec_list = self.__load_db.get_record_list('mitdb')

        while True:
            for (rec, data), anno in zip(self.data_set.items(), self.data_anno.values()):
                anno_sample = np.array(anno['sample'])
                if data.shape[0] > 1*60*self.fs:
                    idx = np.random.randint(0, data.shape[0] - 1*60*self.fs)

                    data = data[idx:idx+1*60*self.fs, :]
                    anno_sample = anno_sample[np.bitwise_and(idx <= anno_sample, anno_sample < idx+1*60*self.fs)] - idx

                col, row = data.shape

                assert col > dim_input, "dim_input can't be greater than data length"

                data = self.pre_processing(data)

                if flip:
                    data = np.flipud(data)

                for i in range(row):
                    temp_data = np.ndarray((col - dim_input, dim_input))
                    temp_anno = np.zeros((col, ))

                    if direction == 'full':
                        for j in range(2*ms_in_sample):
                            if rec in mit_rec_list:
                                anno_idx = np.array(anno_sample) - ms_in_sample + j
                            else:
                                anno_idx = np.array(anno_sample) + j

                            anno_idx[anno_idx >= data.shape[0]] = -1

                            temp_anno[anno_idx[anno_idx >= 0]] = 1
                    elif direction == 'half':
                        for j in range(ms_in_sample):
                            if rec in mit_rec_list:
                                anno_idx = np.array(anno_sample) + j
                            else:
                                anno_idx = np.array(anno_sample) + ms_in_sample + j

                            anno_idx[anno_idx >= data.shape[0]] = -1

                            temp_anno[anno_idx[anno_idx >= 0]] = 1
                    elif direction == 'back_half':
                        for j in range(ms_in_sample):
                            if rec in mit_rec_list:
                                anno_idx = np.array(anno_sample) - j
                            else:
                                anno_idx = np.array(anno_sample) + ms_in_sample - j

                            anno_idx[anno_idx >= data.shape[0]] = -1

                            temp_anno[anno_idx[anno_idx >= 0]] = 1

                    if flip:
                        temp_anno = np.flipud(temp_anno)

                    for j in range(dim_input):
                        temp_data[:, j] = data[j:j-dim_input, i]

                    if augment_on:
                        aug_data = self.data_augmentation(temp_data)
                    else:
                        aug_data = [temp_data]

                    temp_size = temp_data.shape[0]
                    if sparse:
                        temp_anno = np.reshape(
                                    temp_anno[dim_input-1:-1][:timestep_in_sample * (temp_size // timestep_in_sample)],
                                (-1, timestep_in_sample, 1))
                    else:
                        temp_anno = np.reshape(
                                tf.keras.utils.to_categorical(
                                    temp_anno[dim_input-1:-1][:timestep_in_sample * (temp_size // timestep_in_sample)],
                                    num_classes=2),
                                (-1, timestep_in_sample, 2))

                    for temp_data in aug_data:
                        temp_data = np.reshape(
                                temp_data[:timestep_in_sample * (temp_size // timestep_in_sample), :],
                                (-1, timestep_in_sample, dim_input))

                        seg_num = np.cumprod(temp_data.shape)[-1] // (10*250*5*32)
                        if seg_num == 0:
                            iter_list = []
                        else:
                            iter_list = range(np.cumprod(temp_data.shape)[0] // seg_num - 1)

                        for i in iter_list:
                            yield (temp_data[i*seg_num:(i+1)*seg_num], temp_anno[i*seg_num:(i+1)*seg_num])
                        else:
                            yield (temp_data[i*seg_num:], temp_anno[i*seg_num:])


    def create_model(self, dim_input=5):
        model = Sequential()
        model.add(SimpleRNN(5, activation='linear', return_sequences=True, input_shape=(None, dim_input), use_bias=True))
                            #kernel_regularizer=regularizers.l2(5*10**(-4)), bias_regularizer=regularizers.l2(5*10**(-4))))
        model.add(SimpleRNN(7, activation='relu', return_sequences=True, use_bias=True))
                            #kernel_regularizer=regularizers.l2(5*10**(-4)), bias_regularizer=regularizers.l2(5*10**(-4))))
        model.add(SimpleRNN(1, activation='linear', return_sequences=True, use_bias=True))
                            #kernel_regularizer=regularizers.l2(5*10**(-4)), bias_regularizer=regularizers.l2(5*10**(-4))))
        model.add(Activation(activations.relu))
        model.add(Conv1D(1, 25, activation='linear', padding='same', use_bias=True))
                         #kernel_regularizer=regularizers.l2(5*10**(-4)), bias_regularizer=regularizers.l2(5*10**(-4))))
        model.add(Activation(activations.relu))
        model.add(LSTM(64, activation='tanh', return_sequences=True))
        model.add(TimeDistributed(Dense(2, activation='softmax')))

        self.deepqrs_model = model

    def set_model_weight(self, ref_model, num_layers):
        model = self.deepqrs_model

        assert model, "No valid model"

        for i in num_layers:
            model.layers[i].set_weights(ref_model.layers[i].get_weights())

        self.deepqrs_model = model

    def data_augmentation(self, data, dim_input=5):
        data_aug_mult = (-1 if np.random.rand(1)[0] > 0.5 else 1) * (2.5 * np.random.rand(1)[0] + 0.5) * data

        temp_data = np.ndarray((data.shape[0], dim_input))

        random_noise = 100*np.random.rand(1)[0]*np.random.rand(data.shape[0] + dim_input)

        for j in range(dim_input):
            temp_data[:, j] = random_noise[j:j-dim_input]

        data_aug_randn = data + temp_data

        data_single_frequency_noise = 50*np.random.rand(1)[0]*np.sin(2*np.pi*(30.0*np.random.rand(1)[0]) / self.fs * np.arange(data.shape[0] + dim_input) + 2*np.random.rand(1)[0]*np.pi)

        for j in range(dim_input):
            temp_data[:, j] = data_single_frequency_noise[j:j-dim_input]

        data_aug_sin = data + temp_data

        dc_offset = 20 * 200 * (np.random.rand(1)[0] - 0.5)

        data_aug_offset = data + dc_offset

        return [data, data_aug_mult, data_aug_randn, data_aug_sin, data_aug_offset]

    def train_step(self, timestep_per_sample=10000, dim_input=5, direction='full', nepoch=10, augment_on=True, sparse=False, flip=False):
        if not self.data_set:
            self.load_data()

        if not self.deepqrs_model:
            self.create_model(dim_input=dim_input)

        opt = tf.keras.optimizers.Adam(lr=10**(-3))

        if sparse:
            metric = 'sparse_categorical_accuracy'
        else:
            metric = 'categorical_accuracy'

        self.deepqrs_model.compile(optimizer=opt, loss='mse', metrics=[metric])

        self.deepqrs_model.fit_generator(self.get_rnn_data_set(
            dim_input=dim_input,
            timestep=timestep_per_sample,
            direction=direction,
            augment_on=augment_on,
            sparse=sparse,
            flip=flip), steps_per_epoch=2000, epochs=nepoch)

    def train_step_sampling(self, timestep_per_sample=10000, dim_input=5, direction='full', nepoch=10, augment_on=True, sparse=False, flip=False):
        if not self.data_set:
            self.load_data()

        if not self.deepqrs_model:
            self.create_model(dim_input=dim_input)

        opt = tf.keras.optimizers.Adam(lr=10**(-3))

        if sparse:
            metric = 'sparse_categorical_accuracy'
        else:
            metric = 'categorical_accuracy'

        self.deepqrs_model.compile(optimizer=opt, loss='mse', metrics=[metric])

        if augment_on:
            total_sz = 2180
        else:
            total_sz = 436

        self.deepqrs_model.fit_generator(self.get_rnn_data_set_sampling(
            dim_input=dim_input,
            timestep=timestep_per_sample,
            direction=direction,
            augment_on=augment_on,
            sparse=sparse,
            flip=flip), steps_per_epoch=total_sz, epochs=nepoch)

    def get_intermediate_output(self, layer_num, input_data):
        inp = self.deepqrs_model.input
        l = self.deepqrs_model.layers[layer_num]
        out = l.output
        functors = K.function([inp], [out])

        layer_outs = functors([input_data])

        return layer_outs[0]

if __name__ == '__main__':
    inst_deepqrs = deepqrs(load_model('./model/RNN7-RNN5-RNN1-LSTM4-DENSE2.h5'))
    inst_deepqrs.create_model()
    #inst_deepqrs.set_model_weight(load_model('./model/RNN5-RNN7-RNN1-CONV114-LSTM256-DENSE2.h5'), [0, 1, 2])
    inst_deepqrs.train_step_sampling(timestep_per_sample=5000, direction='half', nepoch=30, augment_on=False, sparse=False, flip=False)

    data_set = inst_deepqrs.data_set

    for rec in ['108', 'e0305', '106']:
        dat = data_set[rec]
        dat = inst_deepqrs.make_data_sample(dat[:10*60*250 if dat.shape[0] > 10*60*250 else -1,:])

        out = inst_deepqrs.get_intermediate_output(2, dat[0].reshape((1, dat[0].shape[0], dat[0].shape[1])))

        temp = out[0,:,0]
        temp[temp < 0] = 0
        temp = sp.lfilter(np.ones(50), 1, temp)
        plt.figure()
        ax = plt.subplot(211)
        ax.plot(dat[0][:,-1])
        ax1 = plt.subplot(212, sharex=ax)
        ax1.plot(temp)


# =============================================================================
# mod = inst_deepqrs.deepqrs_model
# w = mod.layers[4].get_weights()
#
# out = inst_deepqrs.get_intermediate_output(3, dat[0].reshape((1, dat[0].shape[0], dat[0].shape[1])))
# out1 = inst_deepqrs.get_intermediate_output(4, dat[0].reshape((1, dat[0].shape[0], dat[0].shape[1])))
#
# =============================================================================
