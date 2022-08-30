import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.losses import *
from tensorflow.keras import *
import tensorflow.keras.backend as K
import numpy as np
import tensorflow.keras as tk
import os


def mlp(classes, number_of_samples, hp):
    input_shape = (number_of_samples)
    input_layer = Input(shape=input_shape, name="input_layer")

    tf.random.set_seed(hp["seed"])

    x = Dense(hp['neurons'], kernel_initializer=hp['weight_init'], activation=hp['activation'], name='fc_1')(input_layer)
    for l_i in range(1, hp["layers"]):
        x = Dense(hp['neurons'], kernel_initializer=hp['weight_init'], activation=hp['activation'], name=f'fc_{l_i + 1}')(x)

    output_layer = Dense(classes, activation='softmax', name=f'output')(x)

    m_model = Model(input_layer, output_layer, name='cnn_softmax')
    optimizer = Adam(lr=hp['learning_rate'])
    m_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    m_model.summary()
    return m_model


def cnn(classes, number_of_samples, hp):
    tf.random.set_seed(hp["seed"])

    input_shape = (number_of_samples, 1)
    input_layer = Input(shape=input_shape, name="input_layer")

    x = Conv1D(hp['filters'], hp['kernel_size'], strides=hp['strides'], activation=hp['activation'], kernel_initializer=hp['weight_init'],
               padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = AveragePooling1D(hp['pool_size'], strides=hp['pool_strides'], padding='same')(x)
    for l_i in range(1, hp["conv_layers"]):
        x = Conv1D(hp['filters'] * (l_i + 1), hp['kernel_size'], strides=hp['strides'], activation=hp['activation'],
                   kernel_initializer=hp['weight_init'], padding='same')(x)
        x = BatchNormalization()(x)
        x = AveragePooling1D(hp['pool_size'], strides=hp['pool_strides'], padding='same')(x)

    x = Flatten()(x)
    for l_i in range(hp["layers"]):
        x = Dense(hp['neurons'], activation=hp['activation'], kernel_initializer=hp['weight_init'], name=f'fc_{l_i + 1}')(x)
    output_layer = Dense(classes, activation='softmax', name=f'output')(x)

    m_model = Model(input_layer, output_layer, name='mlp_softmax')
    optimizer = Adam(lr=hp['learning_rate'])
    m_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    m_model.summary()
    return m_model
