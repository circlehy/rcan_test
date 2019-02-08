import os

from data import common

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class SRData(data.Dataset):
    def __init__(self, args, train=True, benchmark=False):
        self.args = args
        self.train = train
        self.split = 'train' if train else 'test'
        self.benchmark = benchmark
        self.scale = args.scale
        self.idx_scale = 0

        self._set_filesystem(args.dir_data)

        def _load_bin():
            self.images_hr = np.load(self._name_hrbin())
            self.images_hr_map = np.load(self._name_hr_mapbin())
            self.images_lr = [
                np.load(self._name_lrbin(s)) for s in self.scale
            ]
            self.images_map = [
                np.load(self._name_mapbin(s)) for s in self.scale
            ]

        if args.ext == 'img' or benchmark:
            self.images_hr, self.images_lr, self.images_map, self.images_hr_map = self._scan()
        elif args.ext.find('sep') >= 0:
            self.images_hr, self.images_lr, self.images_map, self.images_hr_map = self._scan()
            if args.ext.find('reset') >= 0:
                print('Preparing seperated binary files')
                print (len(self.images_hr))
                for v in self.images_hr:
                    #print v                          #
                    hr = misc.imread(v)
                    name_sep = v.replace(self.ext, '.npy')
                    np.save(name_sep, hr)
                for v in self.images_hr_map:
                    #print v                          #
                    hr_map = misc.imread(v)
                    name_sep = v.replace(self.ext, '.npy')
                    np.save(name_sep, hr_map)
                for si, s in enumerate(self.scale):
                    #print len(self.images_lr[si])    #
                    for v in self.images_lr[si]:
                        #print v                      # 
                        lr = misc.imread(v)
                        name_sep = v.replace(self.ext, '.npy')
                        np.save(name_sep, lr)
                for si, s in enumerate(self.scale):
                    #print len(self.images_lr[si])    #
                    for v in self.images_map[si]:
                        #print v                      # 
                        lr_map = misc.imread(v)
                        name_sep = v.replace(self.ext, '.npy')
                        np.save(name_sep, lr_map)

            self.images_hr = [
                v.replace(self.ext, '.npy') for v in self.images_hr
            ]
            self.images_hr_map = [
                v.replace(self.ext, '.npy') for v in self.images_hr_map
            ]
            self.images_lr = [
                [v.replace(self.ext, '.npy') for v in self.images_lr[i]]
                for i in range(len(self.scale))
            ]
            self.images_map = [
                [v.replace(self.ext, '.npy') for v in self.images_map[i]]
                for i in range(len(self.scale))
            ]

        elif args.ext.find('bin') >= 0:
            try:
                if args.ext.find('reset') >= 0:
                    raise IOError
                print('Loading a binary file')
                _load_bin()
            except:
                print('Preparing a binary file')
                bin_path = os.path.join(self.apath, 'bin')
                if not os.path.isdir(bin_path):
                    os.mkdir(bin_path)

                list_hr, list_lr, list_map, list_hr_map = self._scan()
                hr = [misc.imread(f) for f in list_hr]
                hr_map = [misc.imread(f) for f in list_hr_map]
                np.save(self._name_hrbin(), hr)
                np.save(self._name_hr_mapbin(), hr_map)
                del hr, hr_map
                for si, s in enumerate(self.scale):
                    lr_scale = [misc.imread(f) for f in list_lr[si]]
                    np.save(self._name_lrbin(s), lr_scale)
                    del lr_scale
                for si, s in enumerate(self.scale):
                    map_scale = [misc.imread(f) for f in list_map[si]]
                    np.save(self._name_mapbin(s), map_scale)
                    del map_scale

                _load_bin()
        else:
            print('Please define data type')

    def _scan(self):
        raise NotImplementedError

    def _set_filesystem(self, dir_data):
        raise NotImplementedError

    def _name_hrbin(self):
        raise NotImplementedError

    def _name_lrbin(self, scale):
        raise NotImplementedError

    def _name_mapbin(self, scale):
        raise NotImplementedError


    def __getitem__(self, idx):
        lr, hr, lr_map, hr_map, filename = self._load_file(idx)
        lr, hr, lr_map, hr_map = self._get_patch(lr, hr, lr_map, hr_map)
        lr, hr, lr_map, hr_map = common.set_channel([lr, hr, lr_map, hr_map], self.args.n_colors)
        lr_tensor, hr_tensor, lr_map_tensor, hr_map_tensor = common.np2Tensor([lr, hr, lr_map, hr_map], self.args.rgb_range)
        return lr_tensor, hr_tensor, lr_map_tensor, hr_map_tensor, filename

    def __len__(self):
        return len(self.images_hr)

    def _get_index(self, idx):
        return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        lr = self.images_lr[self.idx_scale][idx]
        lr_map = self.images_map[self.idx_scale][idx]
        hr = self.images_hr[idx]
        hr_map = self.images_hr_map[idx]
        if self.args.ext == 'img' or self.benchmark:
            filename = hr
            lr = misc.imread(lr)
            hr = misc.imread(hr)
            lr_map = misc.imread(lr_map)
            hr_map = misc.imread(hr_map)
        elif self.args.ext.find('sep') >= 0:
            filename = hr
            lr = np.load(lr)
            hr = np.load(hr)
            lr_map = np.load(lr_map)
            hr_map = np.load(hr_map)
        else:
            filename = str(idx + 1)

        filename = os.path.splitext(os.path.split(filename)[-1])[0]

        return lr, hr, lr_map, hr_map, filename

    def _get_patch(self, lr, hr, lr_map, hr_map):
        patch_size = self.args.patch_size
        scale = self.scale[self.idx_scale]
        multi_scale = len(self.scale) > 1
        if self.train:
            lr, hr, lr_map, hr_map = common.get_patch(
                lr, hr, lr_map, hr_map, patch_size, scale, multi_scale=multi_scale
            )
            lr, hr, lr_map, hr_map = common.augment([lr, hr, lr_map, hr_map])
            lr = common.add_noise(lr, self.args.noise)
        else:
            ih, iw = lr.shape[0:2]
            hr = hr[0:ih * scale, 0:iw * scale]
            hr_map = hr_map[0:ih * scale, 0:iw * scale]

        return lr, hr, lr_map, hr_map

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

