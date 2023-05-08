import tensorflow as tf
import keras
import numpy as np
import time
import logging

from keras import layers, Model
from . import BaseClassifier
from .ffnn import FFNN
from src.metrics import get_metrics
from sys import getsizeof

logging.basicConfig(level=logging.INFO)


class MiniAM2Classifier(BaseClassifier):
    _CLASSIFIER_NAME = "miniam2"
    _TEXT_MODELS_NAMES = ["screen_name_ffnn_char", "name_ffnn_char", "name_ffnn_word",
                          "bio_ffnn_char", "bio_ffnn_word"]
    _SCREEN_NAME = "screen_name"
    _NAME = "name"
    _BIO = "bio"
    _TEXT_INPUTS = [_SCREEN_NAME, _NAME, _BIO]
    _TEXT_MODELS_INPUTS = {"screen_name_ffnn_char": _SCREEN_NAME, "name_ffnn_char": _NAME,
                            "name_ffnn_word": _NAME, "bio_ffnn_char": _BIO,
                            "bio_ffnn_word": _BIO}
    _TOK_ARGS_CHAR = {
        "vocabulary_size": 128,
        "input_length": 16,
        "split": "character",
        "output_mode": "int"
    }
    _TOK_ARGS_WORD = {
        "vocabulary_size": 32768,
        "input_length": 16,
        "split": "whitespace",
        "output_mode": "int"
    }
    _TOK_ARGS_WORD_NAME = {
        "vocabulary_size": 32768,
        "input_length": 8,
        "split": "whitespace",
        "variable": "name",
        "output_mode": "int"
    }
    _MODEL_ARGS = {
        "embedding_size": 32,
        "dense_1": 256,
        "dense_2": 64,
        "dropouts": [0.7, 0.7],
    }
    _TEXT_MODEL_TOK_ARGS = {"screen_name_ffnn_char": _TOK_ARGS_CHAR, "name_ffnn_char": _TOK_ARGS_CHAR,
                            "name_ffnn_word": _TOK_ARGS_WORD_NAME, "bio_ffnn_char": _TOK_ARGS_CHAR,
                            "bio_ffnn_word": _TOK_ARGS_WORD}

    def __init__(self):
        super(MiniAM2Classifier, self).__init__(self._CLASSIFIER_NAME)
        self.text_models = None
        self.build_text_models()
        self.build_model()
        self.metrics = {}

    def build_text_models(self):
        self.text_models = {tm_name: FFNN(tm_name) for tm_name in self._TEXT_MODELS_NAMES}
        for tm_name, tm in self.text_models.items():
            tm.build_text_model(self._TEXT_MODEL_TOK_ARGS[tm_name], self._MODEL_ARGS)

    def adapt_text_models_tokenizers(self, data):
        for tm_name, tm in self.text_models.items():
            tm.adapt_tokenizer(data[self._TEXT_MODELS_INPUTS[tm_name]])
        self.build_model()

    def build_model(self):
        """
        Concatenate the texts models, dropout and dense layer. Set the model to 'self.model'
        """
        inp_text = [keras.Input(shape=(1,), dtype=tf.string, name=text_input) for text_input in self._TEXT_INPUTS]

        inp_text_replicate = [tf.identity(inp_text[0]), tf.identity(inp_text[1]),
                              tf.identity(inp_text[1]), tf.identity(inp_text[2]),
                              tf.identity(inp_text[2])]
        text_models_sort = []
        for k, tm_name in enumerate(self._TEXT_MODELS_NAMES):
            FFNN_model = self.text_models.get(tm_name).model
            tm_no_last_layer = keras.Sequential(FFNN_model.layers[:-1])
            text_models_sort.append(tm_no_last_layer(inp_text_replicate[k]))

        concatenation_layer = layers.Concatenate(axis=1)(text_models_sort)
        output = layers.Dense(self.classes, activation="softmax")(concatenation_layer)
        self.model = Model(inp_text, output, name=self.name)

    def warm_up_fit(self, data, label="label", batch=512, epochs=10, test_size=0.1, lr=0.001):
        arr = data[self._TEXT_INPUTS].values
        y = data[label]
        for name, tm in self.text_models.items():
            tm.model.trainable = False
        logging.info("Warm up training...")
        self.fit_in_batches(arr, y, batch=batch, epochs=epochs, test_size=test_size, lr=lr)

    def fit(self, data, label="label", batch=512, epochs=10, test_size=0.1, lr=1e-5, warm_up=True):
        if warm_up:
            self.warm_up_fit(data, label, batch=batch, epochs=epochs, test_size=test_size)
        for name, tm in self.text_models.items():
            tm.model.trainable = True
        arr = data[self._TEXT_INPUTS].values
        y = data[label]
        logging.info("MiniAM2 training...")
        return self.fit_in_batches(arr, y, batch=batch, epochs=epochs, test_size=test_size, lr=lr)

    def fit_text_models(self, data, label="label", batch=512, epochs=10, test_size=0.1, lr=0.001):
        for name, tm in self.text_models.items():
            logging.info(f"Text model {name} training...")
            tm.fit(data, feature=self._TEXT_MODELS_INPUTS[name], label=label, batch=batch,
                   epochs=epochs, test_size=test_size, lr=lr)

    def predict(self, data, batch=512):
        arr = data[self._TEXT_INPUTS].values
        return self.predict_batch(arr, batch)

    def compute_metrics(self, test_sets, label="label"):
        y_trues = {lang: test_set[label] for lang, test_set in test_sets.items()}
        start = time.perf_counter()
        probas = {lang: self.predict(test_set) for lang, test_set in test_sets.items()}
        end = time.perf_counter()
        t = (end - start)
        n = sum([len(test_set) for lang, test_set in test_sets.items()])

        weights = self.model.get_weights()
        mb = np.sum([getsizeof(w) / 1024 / 1024 for w in weights])
        num_params = np.sum([np.prod(w.shape) for w in weights])

        self.metrics["MB"] = mb
        self.metrics["params"] = int(num_params)
        self.metrics["users/s"] = n / t
        self.metrics["s/user"] = t / n

        self.metrics["ovmvw"] = {lang: get_metrics(y, probas[lang], Q='ovmvw') for lang, y in y_trues.items()}
        self.metrics["hvo"] = {lang: get_metrics(y, probas[lang], Q='hvo') for lang, y in y_trues.items()}
        self.metrics["mvw"] = {lang: get_metrics(y, probas[lang], Q='mvw') for lang, y in y_trues.items()}




