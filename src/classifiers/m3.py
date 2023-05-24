import numpy as np
import logging

from m3inference import M3Inference
from . import BaseClassifier

logging.basicConfig(level=logging.INFO)


class M3(BaseClassifier):
    _CLASSIFIER_NAME = "M3"
    _USE_FULL_MODEL = True

    def __init__(self):
        super(M3, self).__init__(self._CLASSIFIER_NAME)
        self.model = M3Inference(use_full_model=self._USE_FULL_MODEL)

    def probas_m3_to_our_probas(self, probas_m3):
        probas = np.zeros((len(probas_m3), 3))
        i = 0
        for key, item in probas_m3.items():
            probas[i, 0] = item["org"]["is-org"]
            human = item["org"]["non-org"]
            probas[i, 1] = item["gender"]["male"] * human
            probas[i, 2] = item["gender"]["female"] * human
            i += 1
        return probas

    def probabilities(self, data, batch_size):
        if type(data) != str:
            index = [str(num) for num in range(len(data))]
            data['id'] = index
            data['id_str'] = index
            data["description"] = data["bio"]
            data = data.to_dict(orient="records")
        logging.info("M3 predicting...")
        p = self.model.infer(data, batch_size=batch_size)
        return p

    def predict(self, df, batch=512):
        probas_m3 = self.probabilities(df, batch_size=batch)
        probas = self.probas_m3_to_our_probas(probas_m3)
        return probas

    def get_model_info(self):
        num_params_m3 = int(np.sum([p.numel() for p in self.model.model.parameters()]))
        mb = np.sum([p.element_size() * p.nelement() / 1024 / 1024 for p in self.model.model.parameters()])
        info = {"Model": self._CLASSIFIER_NAME, "Parameters": int(num_params_m3), "MB": mb}
        return info


class M2(M3):
    _CLASSIFIER_NAME = "M2"
    _USE_FULL_MODEL = False

    def __init__(self):
        super(M2, self).__init__()

