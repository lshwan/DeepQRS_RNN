# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 22:36:20 2020

@author: LSH
"""

import wfdb
import AHADB
import scipy.signal as sp
import numpy as np
from collections import defaultdict

class load_db:
    __db_dir = {'cudb': 'D:\\Database\\CUDB\\',
                'mitdb': 'D:\\Database\\MITDB\\',
                'ahadb': 'D:\\Database\\AHA Database\\',
                'edb': 'D:\\Database\\European ST-T\\',
                'vfdb': 'D:\\Database\\VFDB\\'}
    __beat_anno_list = ['e', 'R', 'n', 'V', '/', 'a', 'Q', 'A', 'f', 'N', 'F', 'E', 'j', 'L', 'S', 'J']
    __non_beat_anno_list = ['x', '|', 'T', '"', '~', '+', '!', '[', ']', 's']
    __rhythm_anno_list = ['(AB', '(AF', '(AFIB', '(AFL', '(ASYS', '(B', '(B3', '(BI', '(BII', '(HGEA', '(IVR', '(N', '(NOD',
                          '(NOISE', '(NSR', '(P', '(PM', '(PREX', '(SAB', '(SBR', '(VER', '(VF', '(VFIB', '(VFL', '(VT', '(T']
    __rhythm_anno_to_human = {'(AB': 'Atrial bigeminy',
                              '(AF': 'Atrial fibrillation',
                              '(AFIB': 'Atrial fibrillation',
                              '(AFL': 'Atrial flutter',
                              '(ASYS': 'Asystole',
                              '(B': 'Ventricular bigeminy',
                              '(B3': '3 deg heart block',
                              '(BI': '1 deg heart block',
                              '(BII': '2 deg heart block',
                              '(HGEA': 'High grade ventricular ectopic activity',
                              '(IVR': 'Idioventricular rhythm',
                              '(N': 'Normal sinus rhythm',
                              '(NOD': 'Nodal rhythm',
                              '(NOISE': 'Noise',
                              '(NSR': 'Normal sinus rhythm',
                              '(P': 'Paced rhythm',
                              '(PM': 'Paced rhythm',
                              '(PREX': 'Pre-excitation',
                              '(SAB': 'Sino-atrial block',
                              '(SBR': 'Sinus bradycardia',
                              '(VER': 'Ventricular escape rhythm',
                              '(VF': 'Ventricular fibraillation',
                              '(VFIB': 'Ventricular fibraillation',
                              '(VFL': 'Ventricular flutter',
                              '(VT': 'Ventricular tachycardia',
                              '(T': 'Ventricular trigeminy'}

    def __init__(self):
        self.__avail_db = ['cudb', 'mitdb', 'ahadb', 'edb', 'vfdb']

    def __overlap_merge__(self, idx_array, start, end):
        end_array = idx_array[:,1]
        end_array[end_array == -1] = 2**32 - 1
        if end == -1:
            end = 2**32 - 1

        ovlp_idx = np.where((start <= idx_array[:,1]) * (end >= idx_array[:,0]))[0]

        if ovlp_idx.shape[0] > 0:
            new_sample = [min(np.min(idx_array[0, ovlp_idx]), start), max(np.max(end_array[ovlp_idx]), end)]
            idx_array = np.delete(idx_array, [2*ovlp_idx, 2*ovlp_idx + 1])
        else:
            new_sample = [start, end]

        if new_sample[1] == 2**32 - 1:
            new_sample[1] = -1

        return np.vstack([np.reshape(idx_array, (-1, 2)), new_sample])

    def __beat_vf_merge__(self, rhythm_dict, beat_type, sample):
        beat_vf_start_idx = np.where(beat_type == '[')[0]
        beat_vf_end_idx = np.where(beat_type == ']')[0]

        if len(rhythm_dict['Ventricular fibraillation']) == 0:
            for i in beat_vf_start_idx:
                start_sample = sample[i]
                end_sample = beat_vf_end_idx[beat_vf_end_idx > start_sample][0] if beat_vf_end_idx[beat_vf_end_idx > start_sample].shape[0] > 0 else -1

                rhythm_dict['Ventricular fibraillation'] = np.vstack([np.reshape(rhythm_dict['Ventricular fibraillation'], (-1, 2)), [start_sample, end_sample]])
        else:
            for i in beat_vf_start_idx:
                start_sample = sample[i]
                end_sample = beat_vf_end_idx[beat_vf_end_idx > start_sample][0] if beat_vf_end_idx[beat_vf_end_idx > start_sample].shape[0] > 0 else -1

                self.__overlap_merge__(rhythm_dict['Ventricular fibraillation'], start_sample, end_sample)

        if len(rhythm_dict['Ventricular fibraillation']) == 0:
            del rhythm_dict['Ventricular fibraillation']

        return rhythm_dict

    def __make_beat_annotation__(self, sample, beat_type):
        beat_anno_samp = [sample[i] for i, b in enumerate(beat_type) if b in self.__beat_anno_list]
        beat_anno = [beat_type[i] for i, b in enumerate(beat_type) if b in self.__beat_anno_list]

        return {'sample': beat_anno_samp, 'beat_type': beat_anno}

    def __make_rhythm_annotation__(self, sample, rhythm_type, beat_type):
        rhythm_type = np.array(rhythm_type)
        beat_type = np.array(beat_type)

        rhythm_idx = np.where(rhythm_type != '')[0]

        rhythm_dict = defaultdict(list)
        start_sample = 0
        cur_rhythm = 'Normal sinus rhythm'

        for i in rhythm_idx:
            if rhythm_type[i].strip() in self.__rhythm_anno_list:
                if start_sample == 0:
                    start_sample = sample[i]
                    cur_rhythm = self.__rhythm_anno_to_human[rhythm_type[i].strip()]
                else:
                    if cur_rhythm != 'Normal sinus rhythm':
                        rhythm_dict[cur_rhythm] = np.vstack([np.reshape(rhythm_dict[cur_rhythm], (-1, 2)), [start_sample, sample[i]]])

                    start_sample = sample[i]
                    cur_rhythm = self.__rhythm_anno_to_human[rhythm_type[i].strip()]
        else:
            if cur_rhythm != 'Normal sinus rhythm':
                rhythm_dict[cur_rhythm] = np.vstack([np.reshape(rhythm_dict[cur_rhythm], (-1, 2)), [start_sample, -1]])

        rhythm_dict = self.__beat_vf_merge__(rhythm_dict, beat_type, sample)

        return rhythm_dict

    def get_record_list(self, db):
        assert db in self.__avail_db, 'Not supported database: %s' %(db)

        if db == 'ahadb':
            return AHADB.get_record_list()
        else:
            return wfdb.get_record_list(db)

    def load_db_data(self, db, record=[], fs=250, gain=200):
        if not record:
            record = self.get_record_list(db)
        else:
            assert db in self.__avail_db, 'Not supported database'

        base_dir = self.__db_dir[db]
        samp = dict()

        for rec in record:
            if db == 'ahadb':
                temp = AHADB.rdsamp(rec)
            else:
                temp = wfdb.rdsamp(base_dir + rec)

            data = temp[0]
            org_fs = temp[1]['fs']
            sig_len = temp[1]['sig_len']

            rsmp_data = data
            rsmp_data[np.isnan(rsmp_data)] = 0;
            if org_fs != fs:
                rsmp_data = sp.resample(data, fs * sig_len // org_fs)

            samp.update({rec: (gain * rsmp_data).astype(int)})

        return samp

    def load_full_db_data(self, dbs, fs=250, gain=200):
        samp = dict()

        try:
            for db in dbs:
                samp.update(self.load_db_data(db, fs=fs, gain=gain))
        except TypeError:
            samp.update(self.load_db_data(dbs, fs=fs, gain=gain))

        return samp

    def load_db_anno(self, db, record=[], fs=250):
        if not record:
            record = self.get_record_list(db)
        else:
            assert db in self.__avail_db, 'Not supported database: %s' %(db)

        base_dir = self.__db_dir[db]

        beat_annotation = dict()
        rhythm_annotation = dict()

        for rec in record:
            if db == 'ahadb':
                temp = AHADB.rdann(rec, 'ano')
            else:
                temp = wfdb.rdann(base_dir + rec, 'atr')

            raw_fs = temp.fs
            anno_sample = temp.sample
            rythm_anno = temp.aux_note
            beat_anno = temp.symbol

            assert raw_fs, 'No sampling frequency %s, %s' %(db, rec)

            anno_sample = (fs * anno_sample) // raw_fs

            beat_annotation.update({rec: self.__make_beat_annotation__(anno_sample, beat_anno)})
            rhythm_annotation.update({rec: self.__make_rhythm_annotation__(anno_sample, rythm_anno, beat_anno)})

        return beat_annotation, rhythm_annotation


    def load_full_db_anno(self, dbs, fs=250):
        beat_annotation = dict()
        rhythm_annotation = dict()

        try:
            for db in dbs:
                beat, rhythm = self.load_db_anno(db, fs=fs)
                beat_annotation.update(beat)
                rhythm_annotation.update(rhythm)

        except TypeError:
            beat, rhythm = self.load_db_anno(dbs, fs=fs)
            beat_annotation.update(beat)
            rhythm_annotation.update(rhythm)

        return beat_annotation, rhythm_annotation


if __name__ == '__main__':
    test = load_db()
    tt = test.get_record_list('mitdb')
    print(tt.__len__())
    tt = test.get_record_list('edb')
    print(tt.__len__())
    tt = test.get_record_list('ahadb')
    print(tt.__len__())

