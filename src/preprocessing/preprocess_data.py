import pandas as pd

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

if __name__ == '__main__':
    preprocess_male_haser()