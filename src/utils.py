import pandas as pd
import logging
import numpy as np
import unidecode
import re
from tqdm import tqdm
import json
from bs4 import *
import requests
from unicodedata import normalize


logging.basicConfig(level=logging.INFO)

ABSTAIN = -1
ORGANIZATION = 0
MAN = 1
WOMAN = 2
HUMAN = 3


def get_data(path, usecols=None, nrows=None):
    # Todo: is it convenient to drop duplicates?
    df = pd.read_csv(path, lineterminator='\n', header=0, na_values=[np.nan, "None ", "None"],
                     usecols=usecols, nrows=nrows).drop_duplicates('screen_name').reset_index(drop=True)
    df = df.fillna('')

    # This is only necessary for test sets:
    if 'label' not in df.columns:
        if 'User_1' in df.columns and 'User_2' in df.columns:
            arr1 = df['User_1'].to_numpy()
            arr2 = df['User_2'].to_numpy()
            if np.sum(arr1 != arr2) > 0:
                logging.info("Different annotations for test set!")
            df['label'] = df['User_1']
    if 'text' not in df.columns:
        df['text'] = df['tweet']
    if 'label' in df.columns:
        df['label'] = df.loc[:, 'label'].apply(lambda x: label_to_int(x))
        df = df.astype({"label": np.int8})
    return df


def label_to_int(x):
    """
    objective: get the labels in integer from string

    Inputs:
        - x, str: the string to consider
    Outputs:
        - int, 0 for orga, 1 for man, 2 for woman
    """
    if x in [ORGANIZATION, MAN, WOMAN, HUMAN, ABSTAIN]:
        return x
    if x == 'Man':
        return MAN
    elif x == 'Woman':
        return WOMAN
    elif x == 'Organization' or x == 'Institution':
        return ORGANIZATION
    elif x == 'Human':
        return HUMAN
    elif x == 'Unknown':
        return ABSTAIN
    else:
        raise Exception("Badly labeled user")


def multiple_replace(text, patterns, substitutions):
    # Create a regular expression  from the dictionary keys
    # regex = re.compile("(%r)" % "|".join(map(re.escape, d.keys())))
    pat = '|'.join(patterns.keys())

    # For each match, look-up corresponding value in dictionary
    return re.sub(pat, lambda mo: search_code(mo.string[mo.start():mo.end()], patterns, substitutions), text)


def preprocess_multiple_substitution(text):
    text = re.sub(r'[!\"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~]', ' ', text)
    return multiple_replace(text, patterns, substitutions)


def search_code(match, patterns, subs):
    for pat, code in patterns.items():
        if match in pat:
            return subs[code]
    logging.info(f"Could not correctly preprocess a character: {match}")
    return "No match"


def preprocess(x, rem_special_characters=True, rem_accents=True, lower=False, norm=False):
    if norm:
        x = normalize('NFKD', x)
    if lower:
        x = x.lower()
    if rem_accents:
        x = remove_accents(x)
    if rem_special_characters:
        x = strip_punctuation(x)
    return x


def find_ngrams(input_list, n):
    return list(zip(*[input_list[i:] for i in range(n)]))


def remove_accents_and_lower(text):
    """
    Objective: Lowercase and finally clear accents
    """
    text = unidecode.unidecode(text)
    text = text.lower()
    # Removes accents.
    cleaned_text = remove_accents(text)
    return cleaned_text


def strip_punctuation(text):
    """
    Objective: Preprocess data; substitute simbols by ' ' and finally clear accents
    """
    # cleaned_text = unidecode.unidecode(text)
    text = " ".join(
        re.findall("[a-zA-ZÀ-ÿ\u0621-\u064A\u0660-\u0669\u0E00-\u0E7F\uac00-\ud7a3\u3040-\u30ff\u4e00-\u9FFF]+", text))
    return text


def remove_accents(text):
    """
    Objective: Lowercase and finally clear accents
    """
    text = unidecode.unidecode(text)
    # Removes accents.
    text = re.sub(u"[àáâãäå]", 'a', text)
    text = re.sub(u"[èéêë]", 'e', text)
    text = re.sub(u"[ìíîï]", 'i', text)
    text = re.sub(u"[òóôõö]", 'o', text)
    text = re.sub(u"[ùúûü]", 'u', text)
    cleaned_text = re.sub(u"[ýÿ]", 'y', text)
    return cleaned_text


lc = {'a': u'[aàáâãäåăąǎǟǡǻȁȃȧ𝐚𝑎𝒂𝓪𝔞𝕒𝖆𝖺𝗮𝘢𝙖𝚊𝝰𝞪]',
      'b': u'[b𝐛𝑏𝓫𝔟𝕓𝖇𝗯𝘣𝙗]',
      'c': u'[cçćĉċčƈ𝐜𝑐𝓬𝕔𝖈𝖼𝗰𝘤𝙘]',
      'd': u'[dďđȡ𝐝𝑑𝓭𝔡𝕕𝖉𝘥𝙙]',
      'e': u'[eèéêëěĕėęȅȇȩɇ𝐞𝑒𝓮𝔢𝕖𝖊𝖾𝗲𝘦𝙚𝒆]',
      'f': u'[fƒ𝐟𝑓𝓯𝔣𝕗𝖋𝖿𝗳𝘧]',
      'g': u'[gĝğġģǥǧǵ𝐠𝑔𝓰𝔤𝕘𝖌𝗀𝗴]',
      'h': u'[hĥȟħ𝐡𝒉𝓱𝕙𝖍𝗁𝗵]',
      'i': u'[iìíîïĩīĭįǐȉȋɨ𝐢𝑖𝒊𝓲𝕚𝖎𝗂𝗶𝘪𝙞𝚒𝒾𝔦]',
      'j': u'[jĵ𝐣𝑗𝓳𝕛𝖏𝗃𝗷]',
      'k': u'[kķǩƙ𝐤𝑘𝓴𝕜𝖐𝗄𝗸]',
      'l': u'[lĺļľŀłƚȴ𝐥𝑙𝓵𝕝𝖑𝗅𝗹]',
      'm': u'[m𝐦𝑚𝓶𝕞𝖒𝗆𝗺]',
      'n': u'[nñńņňǹȵɲ𝐧𝑛𝓷𝕟𝖓𝗇𝒏]',
      'o': u'[oòóôõöøōŏőơǒǫǭȍȏȯȱ𝐨𝑜𝓸𝕠𝖔𝗈𝗼𝘰𝙤𝚘𝜊𝒐]',
      'p': u'[pƥ𝐩𝑝𝓹𝕡𝖕𝗉𝗽]',
      'q': u'[q𝐪𝑞𝓺𝕢𝖖𝗊𝗾]',
      'r': u'[rŕŗřȑȓɍ𝐫𝑟𝓻𝕣𝖗𝗋𝗿𝒓𝓇]',
      's': u'[sśŝşšș]',
      't': u'[tťţŧț𝖙]',
      'u': u'[uùúûüũūŭůűųưǔǖǘǚǜȕȗ𝓾𝖚]',
      'v': u'[vʋ]',
      'w': u'[wŵ]',
      'x': u'[x]',
      'y': u'[yýŷÿȳɏƴ]',
      'z': u'[zźżžƶ]'}

uc = {
    'A': u'[AÀÁÂÃÄÅĂĄǍǞǠǺȀȂȦ𝐀𝑨𝒜𝓐𝔸𝕬𝖠𝗔𝘈𝘼𝙰𝚨𝛢𝜜𝝖𝞐𝟈]',
    'B': u'[B𝐁𝑩𝓑𝔹𝕭𝖡𝗕𝘉𝘽𝙱]',
    'C': u'[CÇĆĈĊČƇ𝐂𝑪𝒞𝓒𝕮𝖢𝗖𝘊𝘾𝙲ᴄ]',
    'D': u'[DĎĐȡ𝐃𝑫𝓓𝔻𝕯𝖣𝗗𝘋𝘿𝙳]',
    'E': u'[EÈÉÊËĚĔĖĘȄȆȨɆƐ𝐄𝑬𝓔𝕰𝖤𝗘𝘌𝙀]',
    'F': u'[FƑ𝐅𝑭𝓕𝔽𝕱𝖥𝗙𝘍𝙁]',
    'G': u'[GĜĞĠĢǤǦǴ𝐆𝑮𝓖𝔾𝕲𝖦𝗚𝘎𝙂]',
    'H': u'[HĤȞĦ𝐇𝑯𝓗𝕳𝖧𝗛𝘏𝙃]',
    'I': u'[IÌÍÎÏĨĪĬĮǏȈȊɪ𝐈𝑰𝓘𝕴𝖨𝗜𝘐𝙄𝚰𝛪𝜤𝝞𝞘]',
    'J': u'[JĴ𝐉𝑱𝓙𝕵𝖩𝗝𝘑𝙅]',
    'K': u'[KĶǨƘ𝐊𝑲𝓚𝕶𝖪𝗞𝘒𝙆]',
    'L': u'[LĹĻĽĿŁȽ𝐋𝑳𝓛𝕷𝖫𝗟𝘓𝙇]',
    'M': u'[M𝐌𝑴𝓜𝕸𝖬𝗠𝘔𝙈]',
    'N': u'[NÑŃŅŇǸȠɲ𝐍𝑵𝓝𝕹𝖭𝗡𝘕𝙉]',
    'O': u'[OÒÓÔÕÖØŌŎŐƠǑǪǬȌȎȮȰ𝐎𝑶𝓞𝕺𝖮𝗢𝘖𝙊𝚶𝛰𝜊𝝤𝞞]',
    'P': u'[PƤ𝐏𝑷𝓟𝕻𝖯𝗣𝘗𝙋]',
    'Q': u'[Q𝐐𝑸𝓠𝕼𝖰𝗤𝘘𝙌]',
    'R': u'[RRŔŖŘȐȒɌ]',
    'S': u'[SŚŜŞŠȘƧ]',
    'T': u'[TŤŢŦȚȾ𝐓𝑻𝓣𝕋𝖳𝗧𝘛𝙏𝓽]',
    'U': u'[UÙÚÛÜŨŪŬŮŰŲƯǓǕǗǙǛȔȖᴜ]',
    'V': u'[Vʋ]',
    'W': u'[WŴ]',
    'X': u'[X]',
    'Y': u'[YÝŶŸȲɎƳ]',
    'Z': u'[ZŹŻŽƵ]'}

patterns = {val: num for num, val in enumerate(list(lc.values()) + list(uc.values()))}
substitutions = {num: val for num, val in enumerate(list(lc.keys()) + list(uc.keys()))}


def abc_dict(string_keys_dic):
    """
    Objective: From a string keys dictionary, create a nested dictionary, where First level keys is the abecedary,
    second level keys are the original keys, abecedary sorted.

    @param string_keys_dic: (dict) A dictionary whose keys are strings
    @return: (dict) A nested dictionary. First level keys is the abecedary, second level keys are the names
    """
    abc_dic = {}
    for key, value in string_keys_dic.items():
        if key[0] in abc_dic:
            abc_dic[key[0]][key] = value
        else:
            abc_dic[key[0]] = {key: value}
    return abc_dic

################################################## No need to have ####################################


def save_dict_as_json(path, dict):
    with open(path, 'w') as f:
        json.dump(dict, f)


def load_json_as_dict(path):
    try:
        f = open(path)
        data = f.read()
        return json.loads(data)
    except Exception as e:
        logging.info("ERROR : " + str(e))
    return False

# Spanish
def compute_dic_names_es(path_names, min_ngrams=4, min_rep=100):
    path_names += '/es/es_names.csv'
    names_es = pd.read_csv(path_names, sep=';', converters={'nombre': str})
    names_es.loc[:, 'nombre'] = names_es.loc[:, 'nombre'].apply(lambda x: int(x.replace('.', '')))
    names_es.loc[:,'preusuel'] = names_es.loc[:,'preusuel'].apply(lambda x: clean_text(x) if type(x) == str else x)
    names_es = names_es.loc[names_es.loc[:, 'preusuel'].apply(lambda x: len(x) if type(x) == str else 0)>1]
    return name_values(names_es, min_ngrams, min_rep)


def compute_dic_hypocoristic_names_es(path_names):
    names_v = load_json_as_dict(path_names)
    if names_v:
        return names_v
    url = "https://es.wikipedia.org/wiki/Anexo:Hipocor%C3%ADsticos_en_espa%C3%B1ol"
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    names_v = {}
    for tag in soup.find_all("li"):
        if tag.i != None:
            hyp = tag.i.find(text=True, recursive=False)
            hyp = clean_text(hyp).split()
            n = tag.find(text=True)
            if '/' in n:
                n = n.split('/')
            else:
                n = [n]
            for n2 in n:
                n2 = clean_text(n2)
                names_v[n2] = hyp
    save_dict_as_json(path_names, names_v)
    return names_v


# French
def compute_dic_names_fr(path_names, min_ngrams=4, min_rep=100):
    path_names += '/fr/nat2020.csv'
    names_fr = pd.read_csv(path_names, sep=';')
    names_fr = names_fr.loc[names_fr.loc[:, "preusuel"] != "_PRENOMS_RARES"]
    names_fr.loc[:,'preusuel'] = names_fr.loc[:,'preusuel'].apply(lambda x: clean_text(x) if type(x) == str else x)
    names_fr = names_fr.loc[names_fr.loc[:, 'preusuel'].apply(lambda x: len(x) if type(x) == str else 0)>1]
    return name_values(names_fr, min_ngrams, min_rep)


def compute_dic_hypocoristic_names_fr(path_names, previous_dic):
    names_v = load_json_as_dict(path_names)
    if not names_v:
        url = "https://fr.wikipedia.org/wiki/Liste_de_pr%C3%A9noms_en_fran%C3%A7ais"
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        names_v = {}
        mf = {'f': -1, 'm': 1}
        for tag in soup.find_all("li"):
            n = tag.text
            if ':' in n and ': origine et signification' not in n and ("(m)" in n or "(f)" in n):
                n = re.sub(r'\sou\s', ' ', n)
                names_info = clean_text(n.split(':')[0]).split()
                gender = names_info[0]
                ns = names_info[1:]
                for n in ns:
                    names_v[n] = mf[gender]
        save_dict_as_json(path_names, names_v)
    names_v = {key: value for key, value in names_v.items() if key not in previous_dic}
    return names_v

# English
def compute_dic_names_en(path_names, min_ngrams=4, min_rep=100):

    names_uk_w = pd.read_csv(path_names + '/en/f_names_uk.csv', sep=';', converters={'Count3': str})
    names_uk_m = pd.read_csv(path_names + '/en/m_names_uk.csv', sep=';', converters={'Count3': str})
    names_us = pd.read_csv(path_names + '/en/us_names.csv', sep=';', converters={'Count3': str})

    # For english language, we need to combine those from uk and us
    names_uk_m.loc[:, 'Count3'] = names_uk_m.loc[:, 'Count3'].apply(
        lambda x: int(x.replace(' ', '').replace('.', '')))
    names_uk_m.loc[:, 'Name'] = names_uk_m.loc[:, 'Name'].apply(lambda x: x[:-1] if x[-1] == ' ' else x)
    names_uk_w.loc[:, 'Count3'] = names_uk_w.loc[:, 'Count3'].apply(
        lambda x: int(x.replace(' ', '').replace('.', '')))
    names_uk_w.loc[:, 'Name'] = names_uk_w.loc[:, 'Name'].apply(lambda x: x[:-1] if x[-1] == ' ' else x)
    names_uk_w.loc[:, 'sexe'] = 2
    names_uk_m.loc[:, 'sexe'] = 1
    names_uk = pd.concat((names_uk_m.loc[:, ['Name', 'Count3', 'sexe']],
                          names_uk_w.loc[:, ['Name', 'Count3', 'sexe']]))
    names_uk.columns = ['preusuel', 'nombre', 'sexe']
    names_us.loc[:, 'sexe'] = names_us.loc[:, 'Gender'].apply(lambda x: 2 if x == 'F' else 1)
    names_us.loc[:, 'Name'] = names_us.loc[:, 'Name'].apply(lambda x: x.upper())
    names_us = names_us.rename(columns={'Frequency': 'nombre', 'Name': 'preusuel'})
    names_en = pd.concat((names_uk, names_us.loc[:, ['preusuel', 'nombre', 'sexe']]))
    names_en.loc[:,'preusuel'] = names_en.loc[:,'preusuel'].apply(lambda x: clean_text(x) if type(x) == str else x)
    names_en = names_en.loc[names_en.loc[:, 'preusuel'].apply(lambda x: len(x) if type(x) == str else 0)>1]
    return name_values(names_en, min_ngrams, min_rep)


def compute_dic_hypocoristic_names_en(path_names):
    names_v = load_json_as_dict(path_names)
    if names_v:
        return names_v
    url = "https://en.wiktionary.org/wiki/Appendix:English_given_names#A"
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    names_v = {}
    for tag in soup.find_all("li"):
        h = tag.find(text=True, recursive=False)
        if tag.a != None and 'title' in tag.a.attrs and h:
            name = tag.a.find(text=True)
            name = clean_text(name)
            short_names = re.sub(r'\sor\s', ' ', h)
            short_names = clean_text(short_names)
            names_v[name] = short_names.split()
    save_dict_as_json(path_names, names_v)
    return names_v

#########################################################################################################
def name_values(names, min_ngrams, min_rep):
    gb = names.groupby(["preusuel", "sexe"])['nombre'].sum()
    man_woman_d = [{}, {}]
    for key, val in tqdm(gb.to_dict().items()):
        name, gender = key
        if val<min_rep and (len(name)<=min_ngrams or len(name.split())>1):
            continue
        man_woman_d[gender-1][name] = val
    d = name_values_from_dicts(man_woman_d[0], man_woman_d[1])
    return d


def relative_numbers(arr_1, arr_2, prior=0):
    """
    Objective: Given two lists of positive integers of same length, get the relative arr_1 to the sum of both lists

    @param arr_1: (array) Names as keys and incidence as values
    @param arr_2: (array) Names as keys and incidence as values
    @param prior: (int) Prior incidence
    @return p: (array) Relative numbers of arr_1 to the sum of both lists
    """

    n, m = len(arr_1), len(arr_2)
    if n != m:
        logging.info("Lists must have the same length")
        return []
    p1 = np.divide(prior + arr_1, 2*prior + arr_1 + arr_2)
    return p1

def name_values_from_dicts(dic_1, dic_2):
    """
    Objective: Build relative numbers of dic_1 compared to the sum of both dictionary values.

    @param dic_1: (dict) Containing names and the absolute numbers
    @param dic_2: (dict) Containing names and the absolute numbers
    @return p_dict: (dict) Relative numbers of dic_1 compared to the sum of both dictionary values
    """

    k1, k2 = dic_1.keys(), dic_2.keys()
    keys = list(set(k1).union(set(k2)))
    keys_length = len(keys)
    arr_1, arr_2 = np.zeros(keys_length), np.zeros(keys_length)
    it = range(keys_length)
    if keys_length > 1e8:
        it = tqdm(it)
    for i in it:
        key = keys[i]
        if key in k1:
            arr_1[i] = dic_1[key]
        if key in k2:
            arr_2[i] = dic_2[key]
    p1 = relative_numbers(arr_1, arr_2)
    p1_dict = dict(zip(keys, p1*2 - 1))
    return p1_dict