'''
Author: G. L. Roberts
Date: May 2019

About: Computer vision project to detect when a driver is distracted
at the wheel.
'''

import os
import numpy
import matplotlib
matplotlib.rcParams['figure.dpi'] = 200
import matplotlib.pyplot as plt
import random
import matplotlib.image as mpimg

CDIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(CDIR, 'data')


def main():
    Explore_Data(print_stats=False, show_ims=True)


class Explore_Data(object):
    def __init__(self, print_stats=False, show_ims=False):
        if print_stats:
            self.count_test_train()
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
