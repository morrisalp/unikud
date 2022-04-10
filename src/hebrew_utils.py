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
# 1466 HEBREW POINT HOLAM HASER FOR VAV
# 1467 HEBREW POINT QUBUTS
# 1468 HEBREW POINT DAGESH OR MAPIQ
# 1469 HEBREW POINT METEG
# 1470 HEBREW PUNCTUATION MAQAF
# 1471 HEBREW POINT RAFE
# 1472 HEBREW PUNCTUATION PASEQ
# 1473 HEBREW POINT SHIN DOT
# 1474 HEBREW POINT SIN DOT
# 1475 HEBREW PUNCTUATION SOF PASUQ
# 1476 HEBREW MARK UPPER DOT
# 1477 HEBREW MARK LOWER DOT
# 1479 HEBREW POINT QAMATS QATAN

NIKUD_START_ORD = 1456
NIKUD_END_ORD = 1479
SPECIAL_ORDS = {1466, 1469, 1470, 1471, 1472, 1475, 1479}

# Extended nikud: includes symbols such as rafe which we strip, but do not add to texts
EXTENDED_NIKUD = {chr(i) for i in range(NIKUD_START_ORD, NIKUD_END_ORD + 1)}
# Nikud: ordinary nikud that we add to texts
NIKUD = {c for c in EXTENDED_NIKUD if ord(c) not in SPECIAL_ORDS}

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