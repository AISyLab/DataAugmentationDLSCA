from tensorflow.keras.callbacks import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class InputGradients(Callback):
    def __init__(self, x_data, y_data, number_of_epochs):
        super().__init__()
        self.current_epoch = 0
        self.x = x_data.astype('float32')
        self.y = y_data
        self.number_of_samples = len(x_data[0])
        self.number_of_epochs = number_of_epochs
        self.gradients = np.zeros((number_of_epochs, self.number_of_samples))

    def on_epoch_end(self, epoch, logs=None):

        input_trace = tf.Variable(self.x)

        with tf.GradientTape() as tape:
            tape.watch(input_trace)
            pred = self.model(input_trace)
            loss = tf.keras.losses.categorical_crossentropy(self.y, pred)

        grad = tape.gradient(loss, input_trace)

        input_gradients = np.zeros(self.number_of_samples)
        for i in range(len(self.x)):
            input_gradients += grad[i].numpy().reshape(self.number_of_samples)

        self.gradients[epoch] = input_gradients

    def get_input_gradients(self):
        for e in range(self.number_of_epochs):
            if np.max(self.gradients[e]) != 0:
                self.gradients[e] = np.abs(self.gradients[e] / np.max(self.gradients[e]))
            else:
                self.gradients[e] = np.abs(self.gradients[e])
        return self.gradients
