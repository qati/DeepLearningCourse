#!/usr/bin/env python
"""
Train resnext of cifar10.

The implementation follows the FAIR github repo:
https://github.com/facebookresearch/ResNeXt
which is slightly different than the arxiv report.
Here I use the non-preactivation blocks.

The specfic settings (lr,batch size ) can be changed 
to follow the original values (with enough GPUs, and time).

Author: Dezso Ribli
"""

CARDINALITY = 16
LR = 0.0125
BATCH_SIZE = 32
EPOCHS_DROP = [150,225]
N_EPOCHS = 250
AUG = True

import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Activation
from keras.layers import BatchNormalization, GlobalAveragePooling2D
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras import optimizers
import math


def resnext(inp, resxt_block, cardinality=4):
    """Return resnext."""
    # inital conv
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(5e-4))(inp)
    x = Activation('relu')(BatchNormalization()(x))
    # residual blocks
    x = resxt_blocks(x, resxt_block, cardinality, 64, 256, 1)
    x = resxt_blocks(x, resxt_block, cardinality, 128, 512, 2)
    x = resxt_blocks(x, resxt_block, cardinality, 256, 1024, 2)
    # classifier
    x = GlobalAveragePooling2D()(x)
    x = Dense(10,activation='softmax')(x)
    model = Model(inputs=inp, outputs=x)
    return model


def resxt_blocks(x, resxt_block, cardinality, n_ch1, n_ch2, init_stride, n_block=3):
    """Perform same size residual blocks."""
    x_shortcut = Conv2D(n_ch2, (1, 1), strides = init_stride, 
                        padding='same', kernel_regularizer=l2(5e-4))(x)
    x_shortcut = BatchNormalization()(x_shortcut)
    # first block
    x = resxt_block(x, x_shortcut, cardinality, n_ch1, n_ch2, init_stride)
    for i in range(n_block-1):  # the other residual blocks
        x = resxt_block(x, x, cardinality, n_ch1, n_ch2)
    return x
         

def resxt_block_a(x, x_shortcut, cardinality, n_ch1, n_ch2, init_stride=1):
    """Perform a residual block."""
    groups=[]
    for i in range(cardinality):
        y = Conv2D(n_ch1, (1, 1), strides=init_stride, 
                   kernel_regularizer=l2(5e-4), padding='same')(x)
        y = Activation('relu')(BatchNormalization()(y))
        y = Conv2D(n_ch1, (3, 3), padding='same', 
                   kernel_regularizer=l2(5e-4),)(y)
        y = Activation('relu')(BatchNormalization()(y))
        y = Conv2D(n_ch2, (1, 1), padding='same', 
                   kernel_regularizer=l2(5e-4),)(y)
        y = BatchNormalization()(y)
        groups.append(y)
    x = keras.layers.add(groups)
    x = keras.layers.add([x, x_shortcut])
    x = Activation('relu')(x)
    return x   


def resxt_block_b(x, x_shortcut, cardinality, n_ch1, n_ch2, init_stride=1):
    """Perform a residual block."""
    groups=[]
    for i in range(cardinality):
        y = Conv2D(n_ch1, (1, 1), strides=init_stride, 
                   kernel_regularizer=l2(5e-4), padding='same')(x)
        y = Activation('relu')(BatchNormalization()(y))
        y = Conv2D(n_ch1, (3, 3), padding='same', 
                   kernel_regularizer=l2(5e-4),)(y)
        y = Activation('relu')(BatchNormalization()(y))
        groups.append(y)
    x = keras.layers.concatenate(groups)
    x = Conv2D(n_ch2, (1, 1), padding='same', 
               kernel_regularizer=l2(5e-4),)(x)
    x = BatchNormalization()(x)
    x = keras.layers.add([x, x_shortcut])
    x = Activation('relu')(x)
    return x   


def norm(x):
    """Normalize images."""
    x = x.astype('float32')
    x[...,0] = (x[...,0] - x[...,0].mean())/x[...,0].std()
    x[...,1] = (x[...,1] - x[...,1].mean())/x[...,1].std()
    x[...,2] = (x[...,2] - x[...,2].mean())/x[...,2].std()
    return x


def step_decay(epoch, base_lr=LR, drop=0.1, epochs_drops=EPOCHS_DROP):
    """Helper for step learning rate decay."""
    lrate = base_lr
    for epoch_drop in epochs_drops:
        lrate *= math.pow(drop,math.floor(epoch/epoch_drop))
        return lrate


if __name__ == "__main__":
    # SGD
    sgd = optimizers.SGD(lr=LR, decay=0, momentum=0.9, nesterov=True)
    
    # resnext
    res = resnext(Input(shape=(32,32,3)), resxt_block_b, CARDINALITY)
    res.compile(loss='sparse_categorical_crossentropy',
                optimizer=sgd, metrics=['accuracy'])
    print res.summary()  # print summary
     
    # load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = map(norm, (x_train, x_test))  # normalize
    
    if AUG:
        # train on generator with standard data augmentation
        gen = ImageDataGenerator(width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 horizontal_flip=True)
        train_generator = gen.flow(x_train, y_train,
                                   batch_size=BATCH_SIZE)
        # train
        res.fit_generator(train_generator, epochs=N_EPOCHS,
                  validation_data=(x_test, y_test),
                  callbacks=[LearningRateScheduler(step_decay)])
    else:
        # just train on data
        res.fit(x_train, y_train, batch_size=BATCH_SIZE, 
                epochs=N_EPOCHS, validation_data=(x_test, y_test),
                callbacks=[LearningRateScheduler(step_decay)])