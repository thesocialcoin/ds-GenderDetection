import numpy as np
import logging
import pandas as pd
import re
from src.utils import preprocess, find_ngrams, MAN, WOMAN, ORGANIZATION, ABSTAIN, abc_dict
from os.path import join
import pickle
from tqdm import tqdm

PROCESSED_DATA_PATH = 'src/classifiers/soft_labelers/data/processed_data'


class LabelingFunctionsModel:
    _CLASS_NAME = "LabelingFunctionsClassifier"

    def __init__(self, lan, processed_data_path=PROCESSED_DATA_PATH, count=False):
        logging.info(f'Creating labeling function model for language: {lan}')
        self.lang = lan
        self.name = self._CLASS_NAME + lan
        self.processed_data_path = processed_data_path
        self.count = count
        self.keywords_hvo = {}
        self.keywords_mvw = {}
        self.lfs_orga = self.get_labeling_functions_hvo()
        self.lfs_mvw = self.get_labeling_functions_mvw()
        self.lfs = self.lfs_orga + self.lfs_mvw

    def apply_soft_label_functions(self, df):
        L = []
        for labeling_function in tqdm(self.lfs):
            logging.info("Applying labeling function to dataset...")
            l = labeling_function(df)
            L.append(l)
            logging.info(f"Abstains: {np.sum(l==ABSTAIN)}     Labels: {np.sum(l!=ABSTAIN)}")
        return np.array(L).T

    def label(self, df, batch=512):
        logging.info('Preprocessing')

        df.loc[:, 'bio_pp'] = df.loc[:, 'bio'].apply(lambda x: preprocess(x, rem_special_characters=False, lower=True))
        df.loc[:, 'text_pp'] = df.loc[:, 'text'].apply(lambda x: preprocess(x, rem_special_characters=False, lower=True))
        df.loc[:, 'name_pp'] = df.loc[:, 'name'].apply(lambda x: preprocess(x, rem_special_characters=True, lower=True))
        df.loc[:, 'screen_name_pp'] = df.loc[:, 'screen_name'].apply(lambda x: preprocess(x, rem_special_characters=False, lower=True))
        df["lang"] = self.lang

        logging.info('Running labelling functions')
        L = self.apply_soft_label_functions(df)
        return L

    def prediction(self, probas, th=0.6):
        return bias_prediction(probas, th)

    def no_bias_prediction(self, probas, th):
        empty_input = np.zeros((1, self.lr_model.layers[0].input_shape[1]))
        bias = self.lr_model.predict(empty_input)
        return no_bias_prediction(probas, bias, th)

    def no_bias_prediction_rescaled(self, probas, th):
        empty_input = np.ones((1, self.lr_model.layers[0].input_shape[1])) * ABSTAIN
        bias = self.lr_model.predict(empty_input)
        return no_bias_prediction_rescaled(probas, bias, th)

    def get_labeling_functions_hvo(self):
        # PeopleHumanKeywords = get_keywords('../src/utils/config')
        path_kewords_lists = join(self.processed_data_path, self.lang)
        count = self.count
        list_names = {'emotions': 'emo_v', 'kin': 'kin_v', 'insults': 'insult_v', 'organization_words': 'orga_v',
                      'institutions': 'name_orga_kw', 'singular_pronouns': 'humpr',
                      'plural_pronouns': 'orgpr', 'social_media': 'social_media', 'url_of_organizations': 'org_url',
                      'man_words': 'man_id_words', 'woman_words': 'woman_id_words'}
        for list_name in list_names:
            with open(join(path_kewords_lists, list_name + '.pkl'), 'rb') as f:
                l = pickle.load(f)
                self.keywords_hvo[list_name] = [preprocess(x, rem_special_characters=False, lower=True) for x in l]
                assert self.keywords_hvo[list_name]

        lf_pro_regex_b = make_keyword_lf_regex('orgpr_b', self.keywords_hvo['plural_pronouns'], df_column="bio_pp", count=count)
        lf_pro_regex_t = make_keyword_lf_regex('orgpr_t', self.keywords_hvo['plural_pronouns'], df_column="text_pp", count=count)
        lf_org_regex_name = make_keyword_lf_regex('orga_v_n', self.keywords_hvo['organization_words'], df_column="name_pp",
                                                  count=count)
        lf_org_regex_bio = make_keyword_lf_regex('orga_v_b', self.keywords_hvo['organization_words'], df_column="bio_pp",
                                                 count=count)
        lf_url_regex = make_sub_keyword_lf_regex('org_url', self.keywords_hvo['url_of_organizations'], df_column="bio", count=count)
        lf_name_org_regex_n = make_keyword_lf_regex('name_orga_kw_n', self.keywords_hvo['institutions'], df_column="name_pp",
                                                    count=count)
        lf_name_org_regex_s = make_sub_keyword_lf_regex('name_orga_kw_s', self.keywords_hvo['institutions'],
                                                        df_column="screen_name_pp", count=count)

        lf_name_in_bio = make_lf_name_in_bio(count=count)
        lf_screen_name_in_bio = make_lf_screen_name_in_bio(count=count)
        lf_has_regex = make_hashtag_lf_regex(name='HashtagsInBio', df_column="bio", label=1, min_occur=2, count=count)
        lf_url_text_regex = make_url_lf_regex(name='UrlsInText', df_column="text", label=1, min_occur=2, count=count)

        # lf_hvo = [lf_prh_text_regex, lf_prh_bio_regex, lf_pro_regex, lf_kin_regex, lf_emo_regex,
        # lf_ins_text_regex, lf_ins_bio_regex, lf_org_regex_name, lf_org_regex_bio, lf_som_regex, lf_url_regex,
        # lf_name_org_regex_n, lf_name_org_regex_s, lf_name_in_bio, lf_screen_name_in_bio, lf_has_regex,
        # lf_url_text_regex]

        lf_hvo = [lf_pro_regex_b, lf_pro_regex_t, lf_org_regex_name, lf_org_regex_bio, lf_url_regex,
                  lf_name_org_regex_n, lf_name_org_regex_s, lf_name_in_bio, lf_screen_name_in_bio,
                  lf_url_text_regex]
        # lf_hvo = [lf_org_regex_name, lf_org_regex_bio, lf_name_org_regex_n, lf_name_org_regex_s]
        lf_hvo = [lf_org_regex_name, lf_org_regex_bio, lf_name_org_regex_n, lf_name_org_regex_s]
        lf_hvo = [lf_pro_regex_b, lf_pro_regex_t]
        lf_hvo = [lf_url_regex, lf_url_text_regex]
        lf_hvo = [lf_name_in_bio, lf_screen_name_in_bio, lf_pro_regex_b, lf_pro_regex_t, lf_org_regex_name,
                  lf_org_regex_bio, lf_name_org_regex_n, lf_name_org_regex_s]
        # lf_hvo = []
        return lf_hvo

    def get_labeling_functions_mvw(self):
        path_kewords_lists = join(self.processed_data_path, self.lang)
        count = self.count
        logging.info(f"Computing {self.lang} dictionary of names")
        with open(join(path_kewords_lists, 'names.pkl'), 'rb') as f:
            l = pickle.load(f)
            self.keywords_mvw['names'] = {preprocess(name, rem_special_characters=False, lower=True): val
                                          for name, val in l.items()}

        # p_abc = abc_dict(p)
        p_ngrams_abc = abc_dict(self.keywords_mvw['names'])

        # h_abc = abc_dict(h)
        # h_ngrams_abc = abc_dict(h_ngrams)
        # with open(join(path_kewords_lists, 'man_words.pkl'), 'rb') as f:
        #     man_words = pickle.load(f)
        # with open(join(path_kewords_lists, 'woman_words.pkl'), 'rb') as f:
        #     woman_words = pickle.load(f)
        # words = make_dict_lookup_words_lf("LookupName", p_abc, 'name_pp', count=count)
        # d_words_h = make_dict_directed_lookup_words_lf("LookupBigLeftNameHypo", h_abc, 'name_pp', count=count)
        # ngrams = make_dict_lookup_ngrams_lf("LookupNgramName", p_abc, 'name_pp', min_ngram=4, count=count)
        # d_ngrams_h = make_dict_directed_lookup_ngrams_lf("LookupBigLeftNgramNameHypo", h_ngrams_abc, 'name_pp',
        # min_ngram=4, count=count)
        # d_words_h_s = make_dict_directed_lookup_words_lf("LookupBigLeftScreenNameHypo", h_abc, 'screen_name_pp', count=count)
        # ngrams_s = make_dict_lookup_ngrams_lf("LookupNgramScreenName", p_abc, 'screen_name_pp', min_ngram=4, count=count)
        # d_ngrams_h_s = make_dict_directed_lookup_ngrams_lf("LookupBigLeftNgramScreenNameHypo", h_ngrams_abc, 'screen_name_pp',
        #                                                  min_ngram=4, count=count)
        # d_words = make_dict_directed_lookup_words_lf("LookupBigLeftName", p_abc, 'name_pp', count=count)
        # d_words_s = make_dict_directed_lookup_words_lf("LookupBigLeftScreenName", p_abc, 'screen_name_pp', count=count)
        # man_kw = make_unpossessed_keyword_lf_regex("ManKw", man_words, "bio_pp", count=count)
        # woman_kw = make_unpossessed_keyword_lf_regex("WomanKw", woman_words, "bio_pp", count=count)

        def man_condition(x):
            return x > 0

        def woman_condition(x):
            return x < 0

        d_ngrams_man = make_dict_directed_lookup_ngrams_lf("LookupBigLeftNgramManName", p_ngrams_abc, 'name_pp',
                                                           man_condition, label=MAN, min_ngram=3, count=count)
        d_ngrams_woman = make_dict_directed_lookup_ngrams_lf("LookupBigLeftNgramWomanName", p_ngrams_abc, 'name_pp',
                                                             woman_condition, label=WOMAN, min_ngram=3, count=count)
        d_ngrams_man_s = make_dict_directed_lookup_ngrams_lf("LookupBigLeftNgramManScreenName", p_ngrams_abc,
                                                             'screen_name_pp',
                                                             man_condition, label=MAN, min_ngram=3, count=count)
        d_ngrams_woman_s = make_dict_directed_lookup_ngrams_lf("LookupBigLeftNgramWomanScreenName", p_ngrams_abc,
                                                               'screen_name_pp',
                                                               woman_condition, label=WOMAN, min_ngram=3, count=count)

        # lfs_mvw = [d_words, d_words_h, d_ngrams, d_ngrams_h, d_words_s, d_words_h_s, d_ngrams_s, d_ngrams_h_s,
        #            man_kw, woman_kw]  # ngrams, ,ngrams_s, words,
        # lfs_mvw = [d_ngrams, d_ngrams_s, man_kw, woman_kw]
        lfs_mvw = [d_ngrams_man, d_ngrams_woman, d_ngrams_man_s, d_ngrams_woman_s]

        # lfs_mvw = [man_kw, woman_kw]
        # lfs_mvw = []
        return lfs_mvw


def train_lr_decision_function(apply_m3_text=True, replicates=10):
    x, y = get_and_prepare_dev_set_for_logistic_training(apply_m3_text)
    no_abstain = (np.sum(x != ABSTAIN, axis=1) != 0)
    x = x[no_abstain]
    y = y[no_abstain]
    x, y = drop_replicates(x, y, replicates)
    logging.info(f"{self.lang} has {len(x)} training sample length")
    self.lr_model = train_lr(x, y)


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


def bias_prediction(probs, th):
    y_pred = np.argmax(probs, axis=1)

    winner_margin = probs[np.arange(len(y_pred)), y_pred]
    abstains = winner_margin < th
    y_pred[abstains] = -1
    return y_pred

def no_bias_prediction(probs, bias, th):
    y_pred = np.argmax(probs, axis=1)

    diffs = probs - bias
    winner_margin = diffs[np.arange(len(y_pred)), y_pred]
    abstains = winner_margin < th
    y_pred[abstains] = -1
    return y_pred


def no_bias_prediction_rescaled(probs, bias, th):
    y_pred = np.argmax(probs, axis=1)

    rescaled_diffs = (probs - bias) / (1-bias)

    winner_margin = rescaled_diffs[np.arange(len(y_pred)), y_pred]

    abstains = winner_margin < th
    y_pred[abstains] = -1
    return y_pred


def pattern_lf_regex(name, pattern, df_column, label, min_occur, count):
    """
    Objective: creat the LF function for snorkel

    Inputs:
        - name, str: the name of the keyword list
        - pattern, str: the pattern to match
        - keywords, list: the list of keywords to look for
        - df_columns, list: the list of columns to concatenate together
        - label, int: the label to tag if match
        - min_occur, int: the number to be above to consider the label
        - count, boolean: to return the number of findings OR ABSTAIN or label
    Outputs:
        - lf snorkel.LF: the label function
    """
    def f(x):
        texts = x[df_column]
        return np.array([pattern_lookup_regex(text, pattern, label, min_occur=1, count=False) for text in texts])
    return f


def make_keyword_lf_regex(name, keywords, df_column, label=ORGANIZATION, count=False):
    assert keywords
    pattern = r'\b(?:{})\b'.format('|'.join(keywords))
    return pattern_lf_regex(name, pattern, df_column, label=label, min_occur=1, count=count)


def make_url_lf_regex(name, df_column, label=ORGANIZATION, min_occur=1, count=False):
    pattern_url = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return pattern_lf_regex(name, pattern_url, df_column, label=label, min_occur=min_occur, count=count)


def make_hashtag_lf_regex(name, df_column, label=ORGANIZATION, min_occur=1, count=False):
    pattern_hashtag = r'(#\w+)\b'
    return pattern_lf_regex(name, pattern_hashtag, df_column, label=label, min_occur=min_occur, count=count)


def make_sub_keyword_lf_regex(name, keywords, df_column, label=ORGANIZATION, count=False):
    assert keywords
    pattern = r'(?:{})'.format('|'.join(keywords))
    return pattern_lf_regex(name, pattern, df_column, label=label, min_occur=1, count=count)


def make_lf_name_in_bio(name="NameInBio", label=ORGANIZATION, count=False):
    def name_in_bio(x):
        """
        Objective: human if the name is in the bio (not in the URL)

        Inputs:
            - x, pd.Series: the row of a pd.DataFrame
            - threshold, boolean: to return ABSTAIN or label OR the number of findings
        Outputs:
            - label, int: if matches otherwise 0
        """
        # Bios building
        pattern_url = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        bios = [re.sub(pattern_url, 'URL', bio) for bio in x['bio_pp'].values]

        # Patterns building
        patterns = [r'(?:{})'.format(n) if len(n) > 5 else r'(?=a)b' for n in x['name_pp'].values]
        return np.array([pattern_lookup_regex(bio, pattern, label=label, count=count)
                         for bio, pattern in zip(bios, patterns)])
    return name_in_bio


def make_lf_screen_name_in_bio(name="ScreenameInBio", label=ORGANIZATION, count=False):
    def screen_name_in_bio(x):
        """
        Objective: human if the name is in the bio (not in the URL)

        Inputs:
            - x, pd.Series: the row of a pd.DataFrame
            - threshold, boolean: to return ABSTAIN or label OR the number of findings
        Outputs:
            - label, int: if matches otherwise 0
        """
        # Bios building
        pattern_url = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        bios = [re.sub(pattern_url, 'URL', bio) for bio in x['bio_pp'].values]

        # Patterns building
        patterns = [r'(?:{})'.format(n) for n in x['screen_name_pp'].values]
        return np.array([pattern_lookup_regex(bio, pattern, label=label, count=count)
                         for bio, pattern in zip(bios, patterns)])
    return screen_name_in_bio


def make_dict_directed_lookup_words_lf(name, d,  variable, threshold=0, count=False, direction='l2r'):
    def f(x):
        return dict_directed_lookup_words_lf(x, d, variable, threshold, count, direction)
    return f


def make_dict_directed_lookup_ngrams_lf(name, d, variable, condition, label, min_ngram=5, count=False):
    def f(df):
        texts = df[variable]
        return np.array([dict_directed_lookup_ngrams_lf(t, d, condition, label, min_ngram, count) for t in texts])
    return f


def make_unpossessed_keyword_lf_regex(name, keywords, df_column, label=1, count=False):
    assert keywords
    pattern = r'\b(?<!mi |la |el |ma |sa |ta |my )(?:{})\b'.format('|'.join(keywords))
    return pattern_lf_regex(name, pattern, df_column, label=label, min_occur=1, count=count)


def pattern_lookup_regex(text, pattern, label, min_occur=1, count=False):
    """
    Objective: from a pattern, in a df_columns, tag as label if you the regex pattern matches in x

    Inputs:
        - x, pd.Series: the row of a pd.DataFrame
        - pattern, str: the pattern to match
        - df_columns, list: the list of columns to concatenate together
        - label, int: the label to tag if match
        - min_occur, int: the number to be above to consider the label
        - count, boolean: to return the number of findings OR ABSTAIN or label
    Outputs:
        - label, int: if matches otherwise 0
    """
    founds = len(re.findall(pattern, text))
    if not count:
        return label if founds >= min_occur else ABSTAIN
    return founds


def dict_directed_lookup_words_lf(x, d, variable, threshold=0., count=False, direction='l2r'):
    """
    Objective: from a name in x and a dic d of names with gender, identify appearing names in dic and give the label MAN,
    WOMAN or ABSTAIN to x by majority vote and a treshold

    Input:
        - x, DataFrame: contains x[variable], the name of a user
        - dic, dict: a dict of probabilities for the name.
        - count, boolean: to return the proba OR ABSTAIN orlabel
    Outputs:
        - , float: the label for name between MAN, WOMAN and ABSTAIN
    """

    assert variable in x.keys()
    text = x[variable]
    if pd.isnull(text):
        return 0
    words = text.split()
    if direction == 'r2l':
        words = words.reverse()
    for word in words:
        decision = d.get(word[0], {}).get(word, 0)
        if decision != 0:
            if count:
                return decision
            else:
                return decision / np.abs(decision)
    return 0


def dict_directed_lookup_ngrams_lf(text, d, condition, label, min_ngram, count):
    """
    Objective: from a name in x and a dic of character ngrams d with gender prop,
                identify appearing names in dic and give the label MAN,
                WOMAN or ABSTAIN to x by majority vote and a treshold

    Input:
        - x, DataFrame: contains x["name"], the name of a user
        - dic, dict: a dict of probabilities for the name.
        - count, boolean: to return the proba OR ABSTAIN orlabel
    Outputs:
        - , float: the label for name between MAN, WOMAN and ABSTAIN
    """
    if pd.isnull(text) or len(text) == 0:
        return ABSTAIN
    n = len(text)
    for i in range(n - min_ngram + 1):
        ngrams = [text[i:j] for j in range(i + min_ngram, n + 1)]
        ngrams.reverse()
        for ngram in ngrams:
            value = d.get(ngram[0], {}).get(ngram, 0)
            if value:
                if condition(value):
                    if count:
                        return value
                    return label
                return ABSTAIN
    return ABSTAIN


