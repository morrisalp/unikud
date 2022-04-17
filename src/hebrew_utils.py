import numpy as np

# Unicode codepoints for nikud:
# NOTE: Some of these are extended nikud which we will not use
# 1456 HEBREW POINT SHEVA
# 1457 HEBREW POINT HATAF SEGOL
# 1458 HEBREW POINT HATAF PATAH
# 1459 HEBREW POINT HATAF QAMATS
# 1460 HEBREW POINT HIRIQ
# 1461 HEBREW POINT TSERE
# 1462 HEBREW POINT SEGOL
# 1463 HEBREW POINT PATAH
# 1464 HEBREW POINT QAMATS
# 1465 HEBREW POINT HOLAM
# 1466 HEBREW POINT HOLAM HASER FOR VAV     ***EXTENDED***
# 1467 HEBREW POINT QUBUTS
# 1468 HEBREW POINT DAGESH OR MAPIQ
# 1469 HEBREW POINT METEG                   ***EXTENDED***
# 1470 HEBREW PUNCTUATION MAQAF             ***EXTENDED***
# 1471 HEBREW POINT RAFE                    ***EXTENDED***
# 1472 HEBREW PUNCTUATION PASEQ             ***EXTENDED***
# 1473 HEBREW POINT SHIN DOT
# 1474 HEBREW POINT SIN DOT

NIKUD_START_ORD = 1456
NIKUD_END_ORD = 1474
SPECIAL_ORDS = {1466, 1469, 1470, 1471, 1472}

# Extended nikud: includes symbols such as rafe which we strip, but do not add to texts
EXTENDED_NIKUD = {chr(i) for i in range(NIKUD_START_ORD, NIKUD_END_ORD + 1)}
# Nikud: ordinary nikud that we add to texts
NIKUD = {c for c in EXTENDED_NIKUD if ord(c) not in SPECIAL_ORDS}

N_VOWELS = len(NIKUD) - 3 # not including dagesh, shin dot, sin dot

idx2chr = dict()
j = 0
for i in range(NIKUD_START_ORD, NIKUD_END_ORD + 1):
    if i not in SPECIAL_ORDS:
        idx2chr[j] = chr(i)
        j += 1

def strip_nikud(s):
    if type(s) is str:
        out = s
        for N in EXTENDED_NIKUD:
            out = out.replace(N, '')
        return out
    out = s.copy() # pd Series
    for N in EXTENDED_NIKUD:
        out = out.str.replace(N, '')
    return out

def text_contains_nikud(text):
    return len(set(text) & EXTENDED_NIKUD) > 0

ABG = set('אבגדהוזחטיכךלמםנןסעפףצץקרשת')

def text_contains_abg(text):
    return len(set(text) & ABG) > 0

# CHARSET = NIKUD | ABG

YUD = 'י'
VAV = 'ו'
YV = YUD + VAV


### utilities for converting (haser, male) text pairs into input & target for nikud model: ###
# haser: includes nikud, but not extra yuds/vavs

def align_haser_male(haser, male):
    '''Input: pairs of texts in ktiv haser (with nikud) and ktiv male
    Output: list of pairs (c1, c2) of characters; c1 in haser, c2 in male'''
    i = 0
    j = 0
    output = []
    while i < len(haser) and j < len(male):
        if i >= len(haser):
            output += [('', male[j])]
            j += 1
        elif j >= len(male):
            output += [(haser[i], '')]
            i += 1
        elif haser[i] == male[j]:
            output += [(haser[i], male[j])]
            i += 1
            j += 1
        elif haser[i] in NIKUD:
            output += [(haser[i], '')]
            i += 1
        else:
            output += [('', male[j])]
            j += 1
            
    return output


def chunk_haser_male(haser, male):
    '''uses alignment from previous method to split text into chunks
    outputs list of chunks, one chunk has format: (str, bool)
    str: Hebrew consonant with vowel(s) attached
    bool: True iff letter should be deleted (i.e. extra yud/vav)'''

    aligned = align_haser_male(haser, male)
    
    chunks = []
    del_flags = []
    cur_chunk = ''
    
    for c1, c2 in aligned:
        if c1 == c2:
            if cur_chunk != '':
                chunks.append(cur_chunk)
                del_flags.append(False)
            cur_chunk = ''
            cur_chunk += c1
        elif c1 == '':
            if cur_chunk != '':
                chunks.append(cur_chunk)
                del_flags.append(False)
            cur_chunk = ''
            chunks.append(c2)
            del_flags.append(True)
        else:
            cur_chunk += c1
    
    if cur_chunk != '':
        chunks.append(cur_chunk)
        del_flags.append(False)
    
    return list(zip(chunks, del_flags))

def chunk2target(chunk):
    '''turns chunks from previous method into multilabel targets for nikud model'''
    
    text, del_flag = chunk
    
    nikkud_list = [
        int(chr(n) in text)
        for n in range(NIKUD_START_ORD, NIKUD_END_ORD + 1)
        if n not in SPECIAL_ORDS
    ]
    
    return nikkud_list + [int(del_flag)]

def haser_male2target(haser, male):
    '''Input: pairs of texts in ktiv haser (with nikud) and ktiv male
    Output: multilabel targets for nikud model'''
    chunked = chunk_haser_male(haser, male)
    return np.vstack([chunk2target(chunk) for chunk in chunked])

if __name__ == '__main__':
    haser = 'הַכְּרֻבִים'
    male = 'הכרובים'
    print(haser)
    print(male)
    print(chunk_haser_male(haser, male))
    print(haser_male2target(haser, male))
    print(haser_male2target(haser, male).shape)