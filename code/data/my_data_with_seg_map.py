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
        #print("AAAA",self.__hash__, self.train)

    def _scan(self):
        #self.train = True #gan ga yi hui
        list_hr = []
        list_hr_map = []
        list_map = [[] for _ in self.scale]
        list_lr = [[] for _ in self.scale]
        #print("BBBB",self.__hash__, self.train)
        if self.train:
            idx_begin = 0
            idx_end = self.args.n_train
        else:
            idx_begin = self.args.n_train
            idx_end = self.args.offset_val + self.args.n_val

        all_file = os.listdir(self.dir_hr)
        print(len(all_file))
        #print(all_file)
        print("idx_begin, idx_end :",idx_begin,idx_end)
        #print(self.train)
        for indx in range(idx_begin,idx_end):
            #filename = '{:0>10}'.format(i)
            #for index_name in filelist:
            print("indx :", indx)
            index_name = all_file[indx]
            if index_name.endswith('png'):
                #print(index_name)
                filename = index_name[0:-4]

                list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
                list_hr_map.append(os.path.join(self.dir_hr_map, filename + self.ext))
                for si, s in enumerate(self.scale):
                    #print(si)
                    list_map[si].append(os.path.join(
                    self.dir_map,
                    'X{}/{}x{}{}'.format(s, filename, s, self.ext)
                    ))
                for si, s in enumerate(self.scale):
                    list_lr[si].append(os.path.join(
                    self.dir_lr,
                    'X{}/{}x{}{}'.format(s, filename, s, self.ext)
                    ))
        print ("len list_hr", len(list_hr))
        print("len list_hr_map", len(list_hr_map))
        print("len list_lr",len(list_lr[0]))
        print("len list_map",len(list_map[0]))
        #print (len(list_lr))
        return list_hr, list_lr, list_map, list_hr_map

    def _set_filesystem(self, dir_data):
        self.apath = dir_data 
        #self.dir_hr = os.path.join(self.apath, '/my_data/my_data_train_HR')
        #self.dir_lr = os.path.join(self.apath, '/my_data/my_data_train_LR_bicubic')
        #self.dir_map = os.path.join(self.apath, '/my_data/my_data_train_LR_map_bicubic')
        self.dir_hr = '/home/hy/workspace/rcan_test/my_data/my_data_train_HR'
        self.dir_lr = '/home/hy/workspace/rcan_test/my_data/my_data_train_LR_bicubic'
        self.dir_map = '/home/hy/workspace/rcan_test/my_data/my_data_train_LR_map_bicubic'
        self.dir_hr_map = '/home/hy/workspace/rcan_test/my_data/my_data_train_HR_map'
        self.ext = '.png'

    def _name_hrbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_HR.npy'.format(self.split)
        )

    def _name_hr_mapbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_HR.npy'.format(self.split)
        )

    def _name_lrbin(self, scale):      
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_LR_X{}.npy'.format(self.split, scale)
        )

    def _name_mapbin(self, scale):
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

