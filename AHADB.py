# -*- coding: utf-8 -*-
"""
Created on Tue May 30 17:56:41 2017

@author: LSH
"""

import numpy as np
import os
import string

def rdsamp(rec):
    """ rdsamp(rec)
    """
    base_dir = 'D:\\Database\\AHA Database\\'
    siglen = 525000
    fs = 250.0
    adcgain = 400.0

    annbase = 5*60*250

    signal_buff = np.ndarray(4*siglen)
    signal = np.ndarray((siglen,2))
    with open(base_dir + rec+'.dat','rb') as f:
        signal_buff = np.array(list(f.read(4*siglen)))
        signal[:,0] = signal_buff[::4] + signal_buff[1::4]  * (2**8)
        signal[:,1] = signal_buff[2::4] + signal_buff[3::4]  * (2**8)

        signal[signal > 2**15] -= 2**16

    return (signal[annbase:,:]/adcgain, {'fs': int(fs), 'n_sig':2, 'sig_len': siglen, 'units': ['mV']})


def rdann(rec, ext):
    """ rdann(rec, ext)
    """
    base_dir = 'D:\\Database\\AHA Database\\'
    f = open(base_dir + rec+ '.' + ext,'rb')

    annolen = os.path.getsize(base_dir + rec + '.' + ext)

    annbase = 0

    annoint = np.array(list(f.read(annolen)), dtype=int)
    anntype = [chr(x) for x in annoint[::6]]

    ann = (annoint[1::6] * (2**16) + annoint[2::6] * (2**8) + annoint[3::6]) / 4 + annbase - 1
    ann = ann.astype('int32')
    f.close()

    class ahadb_annotation:
        sample = []
        symbol = []
        fs = 0
        aux_note = []

        def __init__(self, samp, sym, fs):
            self.sample = samp
            self.symbol = sym
            self.fs = fs
            self.aux_note = [''] * len(samp)

    return ahadb_annotation(ann, anntype, 250)

def get_record_list():
    temp = []
    for i in range(1, 9):
        for j in range(1, 11):
            temp.append(str(i * 1000 + 200 + j))

    return temp
