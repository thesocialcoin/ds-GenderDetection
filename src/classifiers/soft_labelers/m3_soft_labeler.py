import numpy as np
from src.classifiers.m3 import M3, M2
from src.metrics import get_preds_ovmvw


class M3SoftLabeler:
    _CLASSIFIER_NAME = "M3SoftLabeler"
    _USE_FULL_MODEL = True

    def __init__(self):
        if self._USE_FULL_MODEL:
            self.model = M3()

    def label(self, df, batch=512, threshold=0.67):
        probas = self.model.predict(df, batch=batch)
        predicts = get_preds_ovmvw(probas, threshold)
        votes = self.decompose_votes(predicts)
        return votes

    def decompose_votes(self, pred):
        abstains = (pred == -1)
        three_votes = np.zeros((len(pred), 3))
        three_votes[np.arange(len(pred)), pred] = 1
        three_votes[abstains] = np.zeros(3)
        return three_votes


class M2SoftLabeler(M3SoftLabeler):
    _CLASSIFIER_NAME = "M3SoftLabeler"
    _USE_FULL_MODEL = False

    def __init__(self):
        super().__init__()
        self.model = M2()
