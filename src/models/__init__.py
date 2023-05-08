import keras
import logging
import numpy as np
from keras.utils import Sequence
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from keras.callbacks import History, EarlyStopping
from keras.utils.layer_utils import count_params
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)


class BaseClassifier:
    def __init__(self, name, classes=3):
        self.name = name
        self.classes = classes
        self.model = None
        self.history = None

    def load_model_from_path(self, path, version='0.0.0'):
        try:
            model = keras.models.load_model(f'{path}/{self.name}/version/{version}')
            logging.info(f"Trained model found with specified filename: {path}/{self.name}/version/{version}")
            self.model = model
            return True
        except Exception as e:
            logging.info("ERROR : " + str(e))
        return False

    def save_model_to_path(self, path, version, save_format='tf'):
        try:
            self.model.save(f'{path}/{self.name}/version/{version}', save_format=save_format)
        except Exception as e:
            logging.info("ERROR : " + str(e))

    def predict_batch(self, x, batch=512):
        data_batch = BatchGenerator(x.T, y=None, batch_size=batch, shuffle=False)
        return self.model.predict(data_batch)

    def fit_in_batches(self, x, y, batch=512, epochs=10, test_size=0.1, lr=0.001):
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=test_size, random_state=42)

        data_batch = BatchGenerator(x_train.T, y_train, batch_size=batch, shuffle=False)
        val_data_batch = BatchGenerator(x_val.T, y_val, batch_size=batch, shuffle=False)

        history = History()
        cb = [EarlyStopping(patience=3, restore_best_weights=True), history]
        self.model.compile(optimizer=Adam(lr), loss=CategoricalCrossentropy(), metrics=["accuracy"])
        logging.info(f"Total trainable parameters: {count_params(self.model.trainable_weights)}")
        self.history = self.model.fit(data_batch, validation_data=val_data_batch, epochs=epochs, callbacks=cb)


class BatchGenerator(Sequence):
    def __init__(self, x, y, batch_size, shuffle, *kwargs):
        self.x = x
        self.y = y
        self.classes = len(np.unique(self.y))

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.n = self.x.shape[-1]
        self.indexes = np.arange(self.n)
        self.on_epoch_end()

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return (np.ceil(self.n / float(self.batch_size))).astype(int)

    def __getitem__(self, idx):
        ind = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_x = []
        for feature in self.x:
            batch_x.append(feature[ind])
        # v = self.transform(batch_x)
        y_cat = None
        if self.y is not None:
            batch_y = self.y.iloc[ind]
            y_cat = keras.utils.to_categorical(batch_y, num_classes=self.classes, dtype=np.int8)
        return [batch_x, y_cat]

