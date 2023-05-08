import numpy as np
from keras.losses import CategoricalCrossentropy
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, log_loss
from keras.utils import to_categorical
from src.utils import ABSTAIN, MAN, WOMAN, ORGANIZATION, HUMAN


def my_coverage(y_trues, y_preds):
    return np.sum(y_preds != -1) / len(y_trues) * 100


def my_f1_score(y_trues, y_preds):
    return f1_score(y_trues, y_preds, average='macro') * 100


def my_recall_score(y_trues, y_preds):
    return recall_score(y_trues, y_preds, average='macro') * 100


def my_precision_score(y_trues, y_preds):
    return precision_score(y_trues, y_preds, average='macro') * 100


def my_accuracy_score(y_trues, y_preds):
    return accuracy_score(y_trues, y_preds) * 100


def tf_loss(y_trues, y_preds):
    return CategoricalCrossentropy()(y_trues, y_preds)


# "Loss TF"
sd_metrics = ["Accuracy", "F1", "Recall", "Precision", "Coverage", "Loss", "Loss TF"]


def identity(x, threshold=0):
    return x


def get_metrics(y_trues, y_proba, metric_keys=None, Q='ovmvw', threshold=0):
    y_trues = {'ovmvw': y_trues,
               'hvo': np.where(y_trues == ORGANIZATION, ORGANIZATION, HUMAN),
               'mvw': np.where(y_trues != ORGANIZATION, y_trues, ABSTAIN)}[Q]
    labeled = (y_trues != ABSTAIN)
    y_trues = y_trues[labeled]
    y_proba = y_proba[labeled]
    from_proba_to_pred = {'ovmvw': get_preds_ovmvw, 'mvw': get_preds_mvw, 'hvo': get_preds_hvo}[Q]
    y_preds = from_proba_to_pred(y_proba, threshold)
    adapt_proba_Q = {'ovmvw': identity, 'mvw': get_proba_mvw, 'hvo': get_proba_hvo}[Q]
    proba_adapted = adapt_proba_Q(y_proba)
    mets = {}

    if not metric_keys:
        metric_keys = sd_metrics
    if 'Coverage' in metric_keys:
        mets['Coverage'] = my_coverage(y_trues, y_preds)
    y_trues = y_trues[y_preds != ABSTAIN]
    y_preds = y_preds[y_preds != ABSTAIN]
    if 'Accuracy' in metric_keys:
        mets['Accuracy'] = my_accuracy_score(y_trues, y_preds)
    if 'F1' in metric_keys:
        mets['F1'] = my_f1_score(y_trues, y_preds)
    if 'Recall' in metric_keys:
        mets['Recall'] = my_recall_score(y_trues, y_preds)
    if 'Precision' in metric_keys:
        mets['Precision'] = my_precision_score(y_trues, y_preds)
    # if 'Coverage sklearn' in metric_keys:
    #     mets['Coverage sklearn'] = coverage_error(y_trues, y_proba, labels)
    if 'Loss' in metric_keys:
        mets['Loss'] = log_loss(y_trues, proba_adapted)
    if 'Loss TF' in metric_keys:
        classes = np.unique(y_trues)
        max_class = np.max(classes)
        y_cat = to_categorical(y_trues, max_class+1, dtype=np.int8)
        y_cat = y_cat[:, classes]
        mets['Loss TF'] = tf_loss(y_cat, proba_adapted).numpy().astype('float64')
    return mets



def get_abstains(y_proba, threshold):
    y_proba = (y_proba.T / np.sum(y_proba, axis=1)).T
    abstains = (np.max(y_proba, axis=1) < threshold)
    return abstains


def get_preds_ovmvw(y_proba, threshold=0):
    # Normalize and return abstains
    y_pred = np.argmax(y_proba, axis=1)
    abstains = get_abstains(y_proba, threshold)
    y_pred[abstains] = ABSTAIN
    return y_pred


def get_preds_mvw(y_proba, threshold=0):
    # To get the prediction between Man vs Woman, just ignore the probabilities of Orga column
    # False or 0 for woman, True or 1 for man
    sub_proba = np.array([y_proba[:, WOMAN], y_proba[:, MAN]]).T
    y_pred_man = np.argmax(sub_proba, axis=1)
    y_pred_man = np.where(y_pred_man == 1, MAN, WOMAN)

    abstain = get_abstains(sub_proba, threshold)
    y_pred_man[abstain] = ABSTAIN
    return y_pred_man


def get_preds_hvo(y_proba, threshold=0):
    human_proba = y_proba[:, MAN] + y_proba[:, WOMAN]
    organ_proba = y_proba[:, ORGANIZATION]

    # Get the row with 0 if user is a human and 1 if it is an organization
    y_pred_orga = np.argmax([human_proba, organ_proba], axis=0)
    y_pred_orga = np.where(y_pred_orga == 1, ORGANIZATION, HUMAN)

    sub_proba = np.array([human_proba, organ_proba]).T
    abstain = get_abstains(sub_proba, threshold)
    y_pred_orga[abstain] = ABSTAIN

    return y_pred_orga


def get_proba_mvw(y_proba):
    y_proba_adapted = y_proba[:, [MAN, WOMAN]] / (1 - y_proba[:, [ORGANIZATION, ORGANIZATION]])
    return y_proba_adapted


def get_proba_hvo(y_proba):
    y_proba_human = np.array([y_proba[:, ORGANIZATION], y_proba[:, MAN] + y_proba[:, WOMAN]]).T
    return y_proba_human