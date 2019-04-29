# General Imports
import os
import numpy as np
import pandas as pd
from time import strftime, localtime
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':22})

# Deep Learning
import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, TimeDistributed, Flatten, GRU, Dropout, Dense
from keras.models import Sequential
from keras.layers import InputLayer, Conv2D, MaxPooling2D, TimeDistributed, Flatten, GRU, Dropout, Dense,BatchNormalization
import dzr_ml_tf.data_pipeline as dp
from dzr_ml_tf.label_processing import tf_multilabel_binarize
#from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import optimizers

models_list = ['C2_adadelta', 'C1_time', 'C1_frequency', 'C4_adadelta', 'C1_square']


def get_model(model_name ,INPUT_SHAPE ): 
    if model_name == 'C2_square': 
        # Define model architecture
        model = Sequential(
            [
                InputLayer(input_shape=INPUT_SHAPE, name="input_layer"),

                BatchNormalization(),

                Conv2D(activation="relu", filters=32, kernel_size=[3, 3], name="conv_1", padding="same"),
                MaxPooling2D(name="max_pool_1", padding="valid", pool_size=[2, 2]),

                Conv2D(activation="relu", filters=64, kernel_size=[3, 3], name="conv_2", padding="same", use_bias=True),
                MaxPooling2D(name="max_pool_2", padding="valid", pool_size=[2, 2]),

                Flatten(),
                Dense(128, activation='sigmoid', name="dense_1"),
                Dropout(name="dropout_1", rate=0.3),
                Dense(1, activation='sigmoid', name="dense_2"),
            ]
        )
        return model
        
    elif model_name == 'C1_time':
        model = Sequential(
            [
                InputLayer(input_shape=INPUT_SHAPE, name="input_layer"),

                BatchNormalization(),

                Conv2D(activation="relu", filters=32, kernel_size=[1, 60], name="conv_1", padding="same"),
                MaxPooling2D(name="max_pool_1", padding="valid", pool_size=[96, 1]),

                Flatten(),
                Dense(200, activation='sigmoid', name="dense_1"),
                Dropout(name="dropout_1", rate=0.5),
                Dense(1, activation='sigmoid', name="dense_2"),
            ]
        )
        return model
    elif model_name == 'C1_frequency': 
        model = Sequential(
            [
                InputLayer(input_shape=INPUT_SHAPE, name="input_layer"),

                BatchNormalization(),

                Conv2D(activation="relu", filters=32, kernel_size=[32, 1], name="conv_1", padding="same"),
                MaxPooling2D(name="max_pool_1", padding="valid", pool_size=[1, 80]),
                
                Flatten(),
                Dense(200, activation='sigmoid', name="dense_1"),
                Dropout(name="dropout_1", rate=0.5),
                Dense(1, activation='sigmoid', name="dense_2"),
            ]
        )
        return model
        
    elif model_name == 'C4_square':
        # Define model architecture
        model = Sequential(
            [
                InputLayer(input_shape=INPUT_SHAPE, name="input_layer"),

                BatchNormalization(),

                Conv2D(activation="relu", filters=32, kernel_size=[3, 3], name="conv_1", padding="same"),
                MaxPooling2D(name="max_pool_1", padding="valid", pool_size=[2, 2]),

                Conv2D(activation="relu", filters=64, kernel_size=[3, 3], name="conv_2", padding="same", use_bias=True),
                MaxPooling2D(name="max_pool_2", padding="valid", pool_size=[2, 2]),

                Conv2D(activation="relu", filters=128, kernel_size=[3, 3], name="conv_3", padding="same", use_bias=True),
                MaxPooling2D(name="max_pool_3", padding="valid", pool_size=[2, 2]),

                Conv2D(activation="relu", filters=256, kernel_size=[3, 3], name="conv_4", padding="same", use_bias=True),
                MaxPooling2D(name="max_pool_4", padding="valid", pool_size=[2, 2]),
                
                Flatten(),
                Dense(256, activation='sigmoid', name="dense_1"),
                Dropout(name="dropout_1", rate=0.3),
                Dense(1, activation='sigmoid', name="dense_2"),
            ]
        )
        return model
    elif model_name == 'C1_square':
        # Define model architecture
        model = Sequential(
            [
            InputLayer(input_shape=INPUT_SHAPE, name="input_layer"),

            BatchNormalization(),

            Conv2D(activation="relu", filters=32, kernel_size=[12, 12], name="conv_1", padding="same"),
            MaxPooling2D(name="max_pool_1", padding="valid", pool_size=[10,10]),

            Flatten(),
            Dense(128, activation='sigmoid', name="dense_1"),
            Dropout(name="dropout_1", rate=0.5),
            Dense(1, activation='sigmoid', name="dense_2"),
            ]
        )
        return model


def compile_model(model, loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy']):
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

