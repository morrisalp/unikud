import pandas as pd
import unicodedata
from tqdm.auto import tqdm

DATA_DIR = './data'

NIKKUD = {x for x in ''' ֱ  ֲ  ֳ  ִ 
 ֵ  ֶ  ַ  ָ  ֹ 
 ֻ  ּ  ֿ  ׁ  ׂ עְ''' if x not in ' \nע'}
ABG = set('אבגדהוזחטיכךלמםנןסעפףצץקרשת')
CHARSET = NIKKUD | ABG

def strip_nikkud(s):
    if type(s) is str:
        out = s
        for N in NIKKUD:
            out = out.replace(N, '')
        return out
    out = s.copy() # pd Series
    for N in NIKKUD:
        out = out.str.replace(N, '')
    return out

def text_contains_nikkud(text):
    return len(set(text) & NIKKUD) > 0

def text_contains_abg(text):
    return len(set(text) & ABG) > 0

def nikud_slice(text, patience=1):
    words = text.split()
    out = ''
    counter = 0
    on = False
    for w in words:
        if text_contains_nikkud(w) or not text_contains_abg(w):
            on = True
            if len(out) > 0:
                out += ' '
            out += w
        elif on:
            counter += 1
            if counter <= patience:
                out += ' ' + w
            else:
                on = False
                counter = 0
                yield out
                out = ''
    if out != '':
        yield out

def count_max_abg_in_row(text):
    out = 0
    counter = 0
    on = False
    for char in text:
        if char in ABG:
            on = True
            counter += 1
        else:
            out = max(out, counter)
            counter = 0
    return out

def normalize(series):
    tqdm.pandas(desc='Normalizing unicode')
    return series.str.replace(r'\u05ba', '\u05b9', regex=True # "holam haser for vav" => holam
        ).progress_apply(lambda x: unicodedata.normalize('NFC', x)) # splits combining forms (e.g. bet+dagesh)

def preprocess_male_haser():

    print('Preprocessing male-haser data...')

    wiktionary_df = pd.read_csv(f'{DATA_DIR}/raw/he_wiktionary-male_haser.csv'
                            ).drop(columns='haser')
    wikisource_df = pd.read_csv(f'{DATA_DIR}/raw/wikisource-haser_male.csv'
                            ).drop(columns='title'
                            ).rename(columns={'nikkud': 'nikud', 'plain': 'male'})

    wiktionary_df['source'] = 'wiktionary'
    wikisource_df['source'] = 'wikisource'

    df = pd.concat([wiktionary_df, wikisource_df]).fillna('')
    del wiktionary_df, wikisource_df


    df.nikud = normalize(df.nikud)
    df.male = normalize(df.male)

    df.nikud = df.nikud.str.replace(r'\{.*\}', '', regex=True
        ).str.replace(r'\(.*\)', '', regex=True
        ).str.replace(r'[\[\]]', '', regex=True
        ).str.strip()
    df.male = df.male.str.replace(r'[\[\]]', '', regex=True).str.strip()

    df = df[
        df.nikud.notna() &
        df.male.notna() &
        (df.nikud.str.split().str.len() == df.male.str.split().str.len()) &
        (df.male != df.nikud) &
        (~df.nikud.str.contains(r'|', regex=False)) &
        (~df.male.str.contains(r'|', regex=False))
    ]

    word_df = df.set_index('source').apply(lambda x: x.str.split().explode()).reset_index()

    stripped = word_df.nikud.apply(strip_nikkud)

    word_df = word_df[
        ~(stripped.str.endswith('ו') ^ word_df.male.str.endswith('ו')) &
        ~(stripped.str.endswith('י') ^ word_df.male.str.endswith('י')) &
        ~(stripped.str.startswith('ו') ^ word_df.male.str.startswith('ו')) &
        ~(stripped.str.startswith('י') ^ word_df.male.str.startswith('י')) &
        (stripped.str.len() <= word_df.male.str.len())
    ]

    word_df.to_csv(f'{DATA_DIR}/processed/male_haser.csv', index=False)

    print('Done (male-haser)')

def preprocess_nikud_data(nikud_ratio_thresh=0.8, n_words_thresh=3, max_words=50, max_abg_in_row=3):
    print('Preprocessing nikud data...')

    by_series = pd.read_csv(f'{DATA_DIR}/raw/ben-yehuda.txt', header=None)[0]
    wp_series = pd.read_csv(f'{DATA_DIR}/raw/he_wp-nikud.txt', header=None)[0]
    df = pd.DataFrame({
        'text': pd.concat([by_series, wp_series]),
        'source': ['BY'] * by_series.shape[0] + ['WP'] * wp_series.shape[0]
    })
    del by_series
    del wp_series

    df.text = normalize(df.text)

    df = pd.DataFrame([
        {
            'text': S,
            'source': row.source
        }
        for row in tqdm(
            df.sample(df.shape[0]).itertuples(),
            # ^ random shuffle makes progress bar more accurate
            total=df.shape[0], desc='Slicing nikud')
        for S in nikud_slice(row.text)
    ])

    df.text = df.text.str.replace('\u200f', '').str.replace('\xa0', '').str.strip()

    tqdm.pandas(desc='Stripping nikud')
    stripped = df.text.progress_apply(strip_nikkud)

    ratios = stripped.str.len() / df.text.str.len()
    n_words = df.text.str.split().str.len()

    mask = (ratios < nikud_ratio_thresh) & (n_words > n_words_thresh)

    def split_text(text):
        words = text.split(' ')
        out_lists = [[]]
        for w in words:
            if len(out_lists[-1]) >= max_words:
                out_lists.append([])
            out_lists[-1].append(w)
        return [
            ' '.join(L) for L in out_lists
        ]
    
    df = pd.DataFrame([
        {
            'text': T,
            'source': row.source
        }
        for row in tqdm(df[mask].itertuples(), total=mask.sum(), desc='Splitting large texts')
        for T in split_text(row.text)
    ])

    tqdm.pandas(desc='Filtering missing nikkud')
    n_abg_in_row = df.text.progress_apply(count_max_abg_in_row)

    df = df[n_abg_in_row <= max_abg_in_row].copy()

    def rm_last_no_nikud(text):
        last_word = text.split()[-1]
        if last_word != strip_nikkud(last_word):
            return text
        return text[:-len(last_word)].strip()
    
    tqdm.pandas(desc='Removing final words missing nikud')
    df.text = df.text.progress_apply(rm_last_no_nikud)
    df = df[df.text != ''].copy()

    # replace "holam haser for vav" with normal holam
    df.text = df.text.str.replace(r'\u05ba', '\u05b9', regex=True)

    df.to_csv(f'{DATA_DIR}/processed/nikud.csv', index=False)

    print('Done (nikud)')


if __name__ == '__main__':
    preprocess_male_haser()
    preprocess_nikud_data()