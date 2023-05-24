import numpy as np
import logging
from keras import losses, optimizers, layers, Input, Sequential
from . import BaseClassifier


class LR(BaseClassifier):
    _CLASSIFIER_NAME = "LogisticRegression"

    def __init__(self, input_dim, output_dim, name=''):
        super(LR, self).__init__(self._CLASSIFIER_NAME + name)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        inputs = Input(shape=(self.input_dim,), dtype="int8")
        model.add(inputs)
        model.add(layers.Dense(self.output_dim, activation="softmax"))

        model.compile(optimizer=optimizers.Adam(), loss=losses.CategoricalCrossentropy(), metrics=["accuracy"])
        return model

