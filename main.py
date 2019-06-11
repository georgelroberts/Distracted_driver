'''
Author: G. L. Roberts
Date: May 2019

About: Computer vision project to detect when a driver is distracted
at the wheel.

TODO: Build a dataframe/text document to log all previous scores
'''

import os
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import matplotlib.image as mpimg
import pdb
from PIL import Image
import pickle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import load_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import ujson
from sklearn.metrics import log_loss, accuracy_score
matplotlib.rcParams['figure.dpi'] = 200

CDIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(CDIR, 'data')


def main(refit=False, plot_egs=False, cv_scores=False):
    Explore_Data(print_stats=False, show_ims=False)
    train_inst = Load_Data('train', repickle=True)
    data_X, data_y = train_inst.data_X, train_inst.data_y

    train_X, cv_X, train_y, cv_y = train_test_split(
            data_X, data_y, test_size=0.33, random_state=42)
    train_X = train_X / 255
    cv_X = cv_X / 255

    del data_X, data_y
#    mod_fpath = os.path.join(CDIR, 'model1.h5')
    mod_fpath = '.mdl_wts.hdf5'
    model = modelling(train_X[0].shape)
    print(model.summary())
    if refit or not os.path.exists(mod_fpath):
        earlyStopping = EarlyStopping(monitor='val_loss', patience=6,
                                      verbose=0, mode='min')
        mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True,
                                   monitor='val_loss', mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                           patience=2, verbose=1, epsilon=1e-4,
                                           mode='min')
        model.fit(train_X[:, :, :, :], train_y[:, :], epochs=30, verbose=1,
                  batch_size=32,
                  callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
                  validation_data=(cv_X, cv_y))
    model.load_weights(filepath=mod_fpath)

    if plot_egs:
        for i in range(10):
            show_example_prediction(model, cv_X, cv_y)

    if cv_scores:
        pred_log_loss, pred_accuracy = cv_score(model, cv_X, cv_y)
        print("CV log loss: {:.3f}\nCV accuracy: {:.3f}".format(pred_log_loss,
                                                                pred_accuracy))
    prepare_submission(model)


def prepare_submission(model):
    print("Preparing submission")
    test_inst = Load_Data('test', repickle=True)
    test_data = test_inst.data
    test_data = test_data / 255
    sample_sub = pd.read_csv(os.path.join(CDIR, 'sample_submission.csv'))
    preds = model.predict(test_data)
    sub = pd.DataFrame(data=preds,
                       index=sample_sub['img'],
                       columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    sub.reset_index(inplace=True)
    sub.to_csv(os.path.join(CDIR, 'real_sub.csv'), index=False)


def cv_score(model, cv_X, cv_y):
    preds = model.predict(cv_X)
    pred_log_loss = log_loss(cv_y, preds)
    pred_args = np.argmax(preds, axis=1)
    real_args = np.argmax(cv_y, axis=1)
    pred_accuracy = accuracy_score(real_args, pred_args)
    return pred_log_loss, pred_accuracy


def show_example_prediction(model, cv_X, cv_y):
    with open(os.path.join(CDIR, 'classes.json'), 'r') as f:
        class_labels = ujson.load(f)
    idx = np.random.choice(np.arange(len(cv_X)))
    pred = model.predict(cv_X[[idx], :, :, :])
    print(pred)
    pred_cls = np.argmax(pred)
    pred_cls = class_labels[str(pred_cls)]
    actual_cls = np.argmax(cv_y[idx])
    actual_cls = class_labels[str(actual_cls)]
    fig, ax = plt.subplots()
    ax.set_title("Predicted: {}, Actual: {}".format(pred_cls, actual_cls))
    ax.imshow(np.squeeze(cv_X[idx, :, :, :]), cmap='gray')
    plt.tight_layout()
    plt.show()


def modelling(shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))

    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


class Load_Data(object):
    def __init__(self, data, repickle=False):
        self.dataset = data
        self.file = os.path.join(DATA_DIR, '{}_data.pkl'.format(data))
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
            no_test = len(test_lst)
            for ii, test_file in enumerate(test_lst):
                if ii % 500 == 0:
                    print("{:.1f}% complete".format(ii / no_test * 100))
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
            shape_X.append(1)
            self.data_X = self.data_X.reshape(shape_X)
        else:
            shape_X = list(self.data.shape)
            shape_X.append(1)
            self.data = self.data.reshape(shape_X)

    @staticmethod
    def get_label(folder):
        arr = np.zeros(10)
        label_no = int(folder[1])
        arr[label_no] = 1
        return arr

    @staticmethod
    def compress_im(im):
        gs_im = im.convert(mode='L')
        gs_im.thumbnail((200, 200))
        return gs_im


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
    main(refit=True, plot_egs=False)
