# data from http://www-etud.iro.umontreal.ca/~boulanni/icml2012
import os
import numpy as np


class Music_Sequence():

    def __init__(self,
                 which_dataset,
                 forcing_max_length=True,
                 stop=None):

        self.stop = stop
        self.max_label = 88

        if forcing_max_length:
            if which_dataset == 'midi':
                self.max_label = 108
            elif which_dataset == 'nottingham':
                self.max_label = 93
            elif which_dataset == 'muse':
                self.max_label = 105
            elif which_dataset == 'jsb':
                self.max_label = 96

        self.load_data(which_dataset)

    def get_data(self, which_set='train'):
        # Check which_set
        if which_set not in ['train', 'valid', 'test']:
            raise ValueError(which_set + " is not a recognized value. " +
                             "Valid values are ['train', 'valid', 'test'].")
        sequences = self.raw_data[which_set]
        if self.stop is not None:
            assert self.stop <= len(sequences)
            sequences = sequences[:self.stop]        

        X_data = np.asarray(
            [np.asarray([self.list_to_binary_vector(time_step, self.max_label) 
                         for time_step in np.asarray(sequences[i])])
             for i in range(len(sequences))]
        )
                
        return X_data    
        
    def load_data(self, which_dataset):
        # Check which_dataset
        if which_dataset not in ['midi', 'nottingham', 'muse', 'jsb']:
            raise ValueError(which_dataset + " is not a recognized value. " +
                             "Valid values are ['midi', 'nottingham', 'muse', 'jsb'].")

        _data_path = './mdata/'
        if which_dataset == 'midi':
            _path = os.path.join(_data_path + "Piano-midi.de.pickle")
        elif which_dataset == 'nottingham':
            _path = os.path.join(_data_path + "Nottingham.pickle")
        elif which_dataset == 'muse':
            _path = os.path.join(_data_path + "MuseData.pickle")
        elif which_dataset == 'jsb':
            _path = os.path.join(_data_path + "JSB Chorales.pickle")
        self.raw_data = np.load(_path)
     
    def list_to_binary_vector(self, x, dim):
        y = np.zeros((dim,))
        for i in x:
            y[i - 1] = 1
        return y
    
'''
from collections import Iterable
def flatten(items):
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x
'''
