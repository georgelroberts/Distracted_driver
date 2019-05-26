'''
Author: G. L. Roberts
Date: May 2019

About: Computer vision project to detect when a driver is distracted
at the wheel.
'''

import os
import numpy

CDIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(CDIR, 'data')

def main():
    Explore_Data(print_stats=True)


class Explore_Data(object):
    def __init__(self, print_stats=False):
        if print_stats:
            self.count_test_train()

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


if __name__ == '__main__':
    main()

