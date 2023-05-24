import numpy as np
import logging
from . import BaseClassifier
from sys import getsizeof
from keras import utils
logging.basicConfig(level=logging.INFO)
from src.utils import ABSTAIN

class AssemblageDistillation(BaseClassifier):
    _CLASSIFIER_NAME = "AssemblageDistillation"
    _USE_FULL_MODEL = True

    def __init__(self, soft_labelers, decision_function):
        super(AssemblageDistillation, self).__init__(self._CLASSIFIER_NAME)
        self.soft_labelers = soft_labelers
        self.decision_function = decision_function

    def soft_label(self, df, batch=512):
        lab_list = [soft_labeler.label(df, batch) for soft_labeler in self.soft_labelers]
        return np.hstack(lab_list)

    def predict(self, df, batch=512):
        L = self.soft_label(df, batch)
        probas = self.decision_function.predict(L)
        return probas

    def get_model_info(self):
        weights = self.decision_function.get_weights()
        mb = np.sum([getsizeof(w) / 1024 / 1024 for w in weights])
        num_params = np.sum([np.prod(w.shape) for w in weights])
        info = {"Model": self._CLASSIFIER_NAME, "Parameters": int(num_params), "MB": mb}
        return info

    def train_decision_function(self, data, label='label', epochs=300, replicates=10):
        L = self.soft_label(data)
        y = data[label]
        y_cat = utils.to_categorical(y, num_classes=self.classes, dtype=np.int8)
        logging.info("Training decision function...")
        no_abstain = (np.sum(L != ABSTAIN, axis=1) != 0)
        L = L[no_abstain]
        y_cat = y_cat[no_abstain]
        L, y_cat = drop_replicates(L, y_cat, replicates)
        logging.info(f"Training with {len(L)} sample length")
        self.decision_function.model.fit(L, y_cat, epochs=epochs, verbose=0)


def drop_replicates(x, y, replicates=10):
    counter = {}
    indexes = np.zeros(len(x), dtype=bool)
    for i in range(len(x)):
        ele = str(x[i])
        if ele not in counter.keys():
            counter[ele] = 1
            indexes[i] = True
        else:
            if counter[ele] < replicates:
                counter[ele] += 1
                indexes[i] = True
    return x[indexes], y[indexes]