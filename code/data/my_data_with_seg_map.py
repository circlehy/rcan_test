import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc
import os

import torch
import torch.utils.data as data

class my_data_with_seg_map(srdata.SRData):
    def __init__(self, args, train=True):
        super(my_data_with_seg_map, self).__init__(args, train)
        self.repeat = args.test_every // (args.n_train // args.batch_size)

    def _scan(self):
        list_hr = []
        list_map = [[] for _ in self.scale]
        list_lr = [[] for _ in self.scale]
        if self.train:
            idx_begin = 0
            idx_end = self.args.n_train
        else:
            idx_begin = self.args.n_train
            idx_end = self.args.offset_val + self.args.n_val

        all_file = os.walk(self.dir_hr)

        for path_all_file,d,filelist in all_file:
            #filename = '{:0>10}'.format(i)
            for index_name in filelist:
                if index_name.endswith('png'):
                   filename = index_name[0:-4]
                   list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
                   for si, s in enumerate(self.scale):
                       list_map[si].append(os.path.join(
                       self.dir_map,
                       'X{}/{}x{}{}'.format(s, filename, s, self.ext)
                       ))
                   for si, s in enumerate(self.scale):
                       list_lr[si].append(os.path.join(
                       self.dir_lr,
                       'X{}/{}x{}{}'.format(s, filename, s, self.ext)
                       ))

        print (len(list_lr))
        return list_hr, list_lr, list_map

    def _set_filesystem(self, dir_data):
        self.apath = dir_data 
        self.dir_hr = os.path.join(self.apath, 'my_data/my_data_train_HR')
        self.dir_lr = os.path.join(self.apath, 'my_data/my_data_train_LR_bicubic')
        self.dir_map = os.path.join(self.apath, 'my_data/my_data_train_LR_bicubic')
        #self.dir_hr = '/home/hy/workspace/data/my_data/my_data_train_HR'
        #self.dir_lr = '/home/hy/workspace/data/my_data/my_data_train_LR_bicubic'
        self.ext = '.png'

    def _name_hrbin(self):
        
        print('_name_hrbin:',os.path.join(
            self.apath,
            'bin',
            '{}_bin_HR.npy'.format(self.split)
        ))
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_HR.npy'.format(self.split)
        )

    def _name_lrbin(self, scale):
        print('_name_lrbin',os.path.join(
            self.apath,
            'bin',
            '{}_bin_LR_X{}.npy'.format(self.split, scale)
        ))
        
        
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_LR_X{}.npy'.format(self.split, scale)
        )

    def _name_mapbin(self, scale):
        print(os.path.join(self.apath, 'bin', '{}_bin_LR_X{}.npy'.format(self.split, scale)))
        return os.path.join(self.apath, 'bin', '{}_bin_LR_X{}.npy'.format(self.split, scale))


    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

