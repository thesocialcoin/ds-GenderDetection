import pandas as pd
import logging
import numpy as np
import re


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
