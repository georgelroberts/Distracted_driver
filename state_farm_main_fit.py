'''
Author: G. L. Roberts
Date: May 2019

About: Computer vision project to detect when a driver is distracted
at the wheel.

TODO: Build a dataframe/text document to log all previous scores
'''

import os
import shutil
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import matplotlib.image as mpimg
import pdb
import pickle
import ujson
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import load_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from fastai import *
from fastai.vision import *
from fastai.core import *
from sklearn.metrics import log_loss, accuracy_score

from load_data import Load_Data, CDIR, DATA_DIR, MODEL_DIR, RES_DIR
from explore_data import Explore_Data

matplotlib.rcParams['figure.dpi'] = 200



def main(refit=False, plot_egs=False, cv_scores=False):
    keras_inception_transfer()
    # create_val_folder()

    # fastai_fit(refit)
    # with open(os.path.join(RES_DIR, 'saved_res.pkl'), 'rb') as f:
    #     preds = pickle.load(f)
    # preds = preds[0].data.numpy()
    # prepare_submission(preds, fastai=True)

def keras_inception_transfer():
    train_inst = Load_Data('train', repickle=True)
    data_X, data_y = train_inst.data_X, train_inst.data_y
    data_X = data_X / 255.

    data_gen = ImageDataGenerator(vertical_flip=True,
            horizontal_flip=True, height_shift_range=0.1,
            shear_range=0.1, zoom_range=0.1,
            width_shift_range=0.1, preprocessing_function=preprocess_input,
            samplewise_center=True, samplewise_std_normalization=True,
            validation_split=0.2)
    data_gen.fit(data_X)
    batch_size = 32
    train_data_gen = data_gen.flow(data_X, data_y, batch_size=batch_size,
            subset='training')
    valid_data_gen = data_gen.flow(data_X, data_y, batch_size=batch_size,
            subset='validation')

    model = InceptionV3(weights='imagenet', include_top=False,
            input_shape=data_X[0].shape)
    model.trainable = False
    new_model = Sequential()
    new_model.add(model)
    new_model.add(GlobalAveragePooling2D())
    new_model.add(Dropout(0.5))
    new_model.add(Dense(10, activation='softmax'))
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)

    new_model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

    new_model.fit_generator(train_data_gen, epochs=20,
            # steps_per_epoch=train_data_gen // batch_size, 
            validation_data=valid_data_gen,
            # validation_steps=valid_data_gen.samples // batch_size,
            verbose=True)


def fastai_fit(refit):
    data = ImageDataBunch.from_folder('data',
                                      test='test',
                                      valid='valid',
                                      ds_tfms=get_transforms(),
                                      size=224).normalize()
    mod_fname = 'fastai_model'
    mod_fpath = os.path.join(MODEL_DIR, f'{mod_fname}')
    learn = cnn_learner(data, models.resnet34, metrics=accuracy)
    if refit or not os.path.exists(f'{mod_fpath}.pth'):
        learn.unfreeze()
        learn.fit_one_cycle(6, max_lr=slice(1e-6, 1e-4))
        learn.save(mod_fpath)
    else:
        learn = learn.load(mod_fpath)
    preds = learn.get_preds(ds_type=DatasetType.Test)
    with open(os.path.join(RES_DIR, 'saved_res.pkl'), 'wb') as f:
        pickle.dump(preds, f, protocol=2)


def create_val_folder():
    driver_imgs_path = os.path.join(DATA_DIR, 'driver_imgs_list.csv')
    driver_df = pd.read_csv(driver_imgs_path)
    unique_drivers = driver_df.subject.unique()
    np.random.seed(42)
    cv_subj = np.random.choice(unique_drivers, int(len(unique_drivers)/4))
    train_subj = [x for x in unique_drivers if x not in cv_subj]
    val_path = os.path.join(DATA_DIR, 'valid')
    if os.path.exists(val_path):
        return
    os.makedirs(val_path)
    for i in np.linspace(0, 9, 10).astype(int):
        os.makedirs(os.path.join(val_path, f'c{i}'))
    cv_ims =  driver_df[driver_df['subject'].isin(cv_subj)]
    for cls, im_path in zip(cv_ims['classname'], cv_ims['img']):
        orig = os.path.join(DATA_DIR, 'train', cls, im_path)
        new = os.path.join(DATA_DIR, 'valid', cls, im_path)
        shutil.move(orig, new)


def load_and_fit(refit, plot_egs, cv_scores):
    train_inst = Load_Data('train', repickle=False)
    data_X, data_y = train_inst.data_X, train_inst.data_y

    train_X, cv_X, train_y, cv_y = train_test_split(
            data_X, data_y, test_size=0.33, random_state=42)
    train_X = train_X / 255
    cv_X = cv_X / 255

    del data_X, data_y
    mod_fpath = os.path.join(MODEL_DIR, '.mdl_wts.hdf5')
    model = modelling(train_X[0].shape)
    if refit or not os.path.exists(mod_fpath):
        earlyStopping = EarlyStopping(monitor='val_loss', patience=6,
                                      verbose=0, mode='min')
        mcp_save = ModelCheckpoint(mod_fpath, save_best_only=True,
                                   monitor='val_loss', mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                           patience=2, verbose=1, epsilon=1e-4,
                                           mode='min')
        model.fit(train_X[:, :, :, :], train_y[:, :], epochs=100, verbose=1,
                  batch_size=32,
                  callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
                  validation_data=(cv_X, cv_y))
    model.load_weights(filepath=mod_fpath)

    if plot_egs:
        for i in range(10):
            show_example_prediction(model, cv_X, cv_y)

    if cv_scores:
        pred_log_loss, pred_accuracy = cv_score(model, cv_X, cv_y)
        print(f"CV log loss: {pred_log_loss:.3f}\nCV accuracy: {pred_accuracy:.3f}")
    test_inst = Load_Data('test', repickle=False)
    test_data = test_inst.data
    test_data = test_data / 255
    preds = model.predict(test_data)
    prepare_submission(preds)


def prepare_submission(preds, fastai=False):
    print("Preparing submission")
    if fastai:
        idxs = os.listdir(os.path.join(DATA_DIR, 'test'))
    else:
        sample_sub = pd.read_csv(os.path.join(RES_DIR, 'sample_submission.csv'))
        idxs = sample_sub['img']
    sub = pd.DataFrame(data=preds,
                       index=idxs,
                       columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    sub.index.name = 'img'
    sub.reset_index(inplace=True)
    sub.to_csv(os.path.join(RES_DIR, 'real_sub.csv'), index=False)


def cv_score(model, cv_X, cv_y):
    preds = model.predict(cv_X)
    pred_log_loss = log_loss(cv_y, preds)
    pred_args = np.argmax(preds, axis=1)
    real_args = np.argmax(cv_y, axis=1)
    pred_accuracy = accuracy_score(real_args, pred_args)
    return pred_log_loss, pred_accuracy


def show_example_prediction(model, cv_X, cv_y):
    with open(os.path.join(DATA_DIR, 'classes.json'), 'r') as f:
        class_labels = ujson.load(f)
    idx = np.random.choice(np.arange(len(cv_X)))
    pred = model.predict(cv_X[[idx], :, :, :])
    print(pred)
    pred_cls = np.argmax(pred)
    pred_cls = class_labels[str(pred_cls)]
    actual_cls = np.argmax(cv_y[idx])
    actual_cls = class_labels[str(actual_cls)]
    fig, ax = plt.subplots()
    ax.set_title(f"Predicted: {pred_cls}, Actual: {actual_cls}")
    ax.imshow(np.squeeze(cv_X[idx, :, :, :]), cmap='gray')
    plt.tight_layout()
    plt.show()


def modelling(shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model



if __name__ == '__main__':
    main(refit=False, plot_egs=False, cv_scores=False)
