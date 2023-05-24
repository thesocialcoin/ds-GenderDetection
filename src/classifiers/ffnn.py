import tensorflow as tf
from tensorflow import keras
import logging
import numpy as np
from sys import getsizeof
from . import BaseClassifier

logging.basicConfig(level=logging.INFO)


class FFNN(BaseClassifier):
    _CLASSIFIER_NAME = "FFNN"
    _MODEL_ARGS_NEEDED = ["embedding_size", "dense_1", "dense_2", "dropouts"]
    _TOK_ARGS_NEEDED = ["vocabulary_size", "input_length", "split", "output_mode"]
    _TOKENIZER_NAME = _CLASSIFIER_NAME + "_TOKENIZER"
    _EMBEDDING_NAME = _CLASSIFIER_NAME + "_EMBEDDING"

    def __init__(self, text_input):
        super(FFNN, self).__init__(self._CLASSIFIER_NAME + "_" + text_input)
        self.build_model = self.build_text_model
        self.text_input = text_input
        self.tokenizer_args = None
        self.model_args = None
        self.tokenizer = None

    def build_tokenizer(self, tokenizer_args):
        if not self.verify_tokenizer_args(tokenizer_args):
            logging.info(f"Missing args to build FFNN tokenizer of required {self._TOK_ARGS_NEEDED}")
            return False

        self.tokenizer_args = tokenizer_args

        standardize = 'lower_and_strip_punctuation'
        if self.tokenizer_args["split"] == 'character':
            standardize = None
        tokenizer = keras.layers.TextVectorization(max_tokens=self.tokenizer_args["vocabulary_size"],
                                                   output_mode=self.tokenizer_args["output_mode"],
                                                   output_sequence_length=self.tokenizer_args["input_length"],
                                                   split=self.tokenizer_args["split"],
                                                   standardize=standardize,
                                                   name=self._TOKENIZER_NAME)
        self.tokenizer = tokenizer
        return True

    def adapt_tokenizer(self, data):
        max_batches_to_adapt = 200
        n = len(data)
        b = n // max_batches_to_adapt
        logging.info("Adapting vocabularies...")
        self.tokenizer.adapt(data, batch_size=b)
        logging.info("Adapted.")

    def build_text_model(self, tokenizer_args, model_args):
        success = self.build_tokenizer(tokenizer_args)
        if not success:
            logging.info("Failed building the tokenizer")
            return
        success = self.build_ffnn_model(model_args)
        if not success:
            logging.info("Failed building the model")
            return

    def build_ffnn_model(self, model_args):
        if not self.verify_model_args(model_args):
            logging.info(f"Missing args to build FFNN tokenizer of required {self._MODEL_ARGS_NEEDED}")
            return False

        self.model_args = model_args
        # Build the NLP model
        model = keras.Sequential(name=self.name)
        inp = keras.Input(shape=(1,), dtype=tf.string)

        model.add(inp)
        model.add(self.tokenizer)

        # Embedding layer
        model.add(keras.layers.Embedding(self.tokenizer_args["vocabulary_size"],
                                         self.model_args["embedding_size"],
                                         input_length=self.tokenizer_args["input_length"],
                                         trainable=True,
                                         name=self._EMBEDDING_NAME))

        # FFNN
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(model_args["dense_1"], activation='relu', kernel_initializer="glorot_uniform"))
        model.add(keras.layers.Dropout(model_args["dropouts"][0]))
        model.add(keras.layers.Dense(model_args["dense_2"], activation='relu', kernel_initializer="glorot_uniform"))
        model.add(keras.layers.Dropout(model_args["dropouts"][1]))

        model.add(keras.layers.Flatten())

        # Add a classifier
        model.add(keras.layers.Dense(self.classes, activation="softmax"))
        model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.CategoricalCrossentropy(),
                      metrics=["accuracy"])
        self.model = model
        return True

    def predict(self, data, batch=512):
        col_arr = data[self.text_input].values
        arr = col_arr.reshape(len(col_arr), 1)
        return self.predict_batch(arr, batch)

    def fit(self, data, feature, label="label", batch=512, epochs=10, test_size=0.1, lr=0.001):
        col_arr = data[feature].values
        arr = col_arr.reshape(len(col_arr), 1)
        y = data[label]

        return self.fit_in_batches(arr, y, batch=batch, epochs=epochs, test_size=test_size, lr=lr)

    def verify_tokenizer_args(self, args):
        args_names = list(args.keys())
        return set(self._TOK_ARGS_NEEDED) <= set(args_names)

    def verify_model_args(self, args):
        args_names = list(args.keys())
        return set(self._MODEL_ARGS_NEEDED) <= set(args_names)

    def get_model_info(self):
        weights = self.model.get_weights()
        mb = np.sum([getsizeof(w) / 1024 / 1024 for w in weights])
        num_params = np.sum([np.prod(w.shape) for w in weights])
        info = {"Model": self.name, "Parameters": int(num_params), "MB": mb}
        return info
