'''
Author: G. L. Roberts
Date: May 2019

About: Computer vision project to detect when a driver is distracted
at the wheel.
'''

import os
import numpy
import matplotlib
import numpy as np
matplotlib.rcParams['figure.dpi'] = 200
import matplotlib.pyplot as plt
import random
import matplotlib.image as mpimg
from PIL import Image

CDIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(CDIR, 'data')


def main():
    Explore_Data(print_stats=False, show_ims=False)
    Load_Data('train')


class Load_Data(object):
    def __init__(self, data, repickle=False):
        self.dataset = data
        self.file = os.path.join(DATA_DIR, '{}_data.pkl'.format(data))
        if not os.path.exists(self.file) or repickle:
            self.build_input()
        else:
            self.load_input()

    def build_input(self):
        if self.dataset == 'train':
            train_dir = os.path.join(DATA_DIR, 'train')
            train_folds = os.listdir(train_dir)
            train_data = []
            for fold in train_folds:
                class_dir = os.path.join(train_dir, fold)
                im_paths = os.listdir(class_dir)
                for im_path in im_paths:
                    im = Image.open(os.path.join(class_dir, im_path))
                    train_data.append([np.array(im), fold])

        elif self.dataset == 'test':
            test_lst = os.listdir(os.path.join(DATA_DIR, 'test'))
            test_data = []
            for test_file in test_lst:
                fpath = os.path.join(DATA_DIR, 'test', test_file)
                im = Image.open(fpath)
                test_data.append(np.array(im))

    def load_input(self):
        pass


class Explore_Data(object):
    def __init__(self, print_stats=False, show_ims=False):
        if print_stats:
            self.count_test_train()
            self.unique_im_sizes()
        if show_ims:
            self.show_class_examples()

    def count_test_train(self):
        train_lst = os.listdir(os.path.join(DATA_DIR, 'train'))
        print("No train classes: {}".format(len(train_lst)))
        tot_train = 0
        for ii, fold in enumerate(train_lst):
            class_lst = os.listdir(os.path.join(DATA_DIR, 'train', fold))
            print("No. examples in fold {}: {}".format(ii, len(class_lst)))
            tot_train += len(class_lst)
        print("No train images: {}".format(tot_train))

        test_lst = os.listdir(os.path.join(DATA_DIR, 'test'))
        print("No test images: {}".format(len(test_lst)))

    def unique_im_sizes(self):
        train_dir = os.path.join(DATA_DIR, 'train')
        train_folds = os.listdir(train_dir)
        sizes = set()
        for fold in train_folds:
            class_dir = os.path.join(train_dir, fold)
            im_paths = os.listdir(class_dir)
            for im_path in im_paths:
                im = Image.open(os.path.join(class_dir, im_path))
                width, height = im.size
                channels = len(im.mode)
                sizes.add((width, height, channels))
        print("Unique image sizes: {}".format(sizes))

    def show_class_examples(self):
        train_dir = os.path.join(DATA_DIR, 'train')
        train_folds = os.listdir(train_dir)
        fig, axes = plt.subplots(2, 5, figsize=(20, 10))
        for ax, train_set in zip(axes.reshape(-1), train_folds):
            im_paths = os.listdir(os.path.join(train_dir, train_set))
            im_path = random.choice(im_paths)
            img = mpimg.imread(os.path.join(train_dir, train_set, im_path))
            ax.set_title("Train set: {}".format(train_set))
            ax.imshow(img)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
