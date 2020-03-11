'''
Author: G. L. Roberts
Date: March 2020

About: Load/manipulate data from way presented in kaggle

'''

import os
import pickle
import numpy as np
import pdb
from PIL import Image


CDIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(CDIR, 'data')
MODEL_DIR = os.path.join(CDIR, 'saved_models')
RES_DIR = os.path.join(CDIR, 'results')

class Load_Data(object):
    def __init__(self, data, repickle=False):
        self.dataset = data
        self.file = os.path.join(DATA_DIR, f'{data}_data.pkl')
        if not os.path.exists(self.file) or repickle:
            self.build_input()
        else:
            self.load_input()
        self.manipulate_input()

    def build_input(self):
        self.data = []
        if self.dataset == 'train':
            train_dir = os.path.join(DATA_DIR, 'train')
            train_folds = os.listdir(train_dir)
            for fold in train_folds:
                print(fold)
                label = self.get_label(fold)
                class_dir = os.path.join(train_dir, fold)
                im_paths = os.listdir(class_dir)
                for im_path in im_paths:
                    im = Image.open(os.path.join(class_dir, im_path))
                    im = self.compress_im(im)
                    self.data.append([np.array(im), label])

        elif self.dataset == 'test':
            test_lst = os.listdir(os.path.join(DATA_DIR, 'test'))
            test_lst.sort()
            no_test = len(test_lst)
            for ii, test_file in enumerate(test_lst):
                if ii % 500 == 0:
                    print("{(ii / no_test * 100):.1f}% complete")
                fpath = os.path.join(DATA_DIR, 'test', test_file)
                im = Image.open(fpath)
                im = self.compress_im(im)
                self.data.append(np.array(im))
        self.data = np.array(self.data)
        with open(self.file, 'wb') as f:
            pickle.dump(self.data, f, protocol=2)

    def load_input(self):
        with open(self.file, 'rb') as f:
            self.data = pickle.load(f)
        self.data = np.array(self.data)

    def manipulate_input(self):
        if self.dataset == 'train':
            np.random.shuffle(self.data)
            self.data_X = np.array(list(self.data[:, 0]))
            self.data_y = np.array(list(self.data[:, 1]))
            shape_X = list(self.data_X.shape)
            # shape_X.append(1)
            self.data_X = self.data_X.reshape(shape_X)
        else:
            shape_X = list(self.data.shape)
            # shape_X.append(1)
            self.data = self.data.reshape(shape_X)

    @staticmethod
    def get_label(folder):
        arr = np.zeros(10)
        label_no = int(folder[1])
        arr[label_no] = 1
        return arr

    @staticmethod
    def compress_im(im):
        # gs_im = im.convert(mode='L')
        im.thumbnail((256, 256, 3))
        return im


