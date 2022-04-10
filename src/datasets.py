import torch
from torch.utils.data import Dataset
import pandas as pd
from hebrew_utils import NIKUD, strip_nikud, YUD, VAV, YV

def make_male_labels(text_n, text_non):
    out = ''
    j = -1
    for i, c in enumerate(text_n):
        if c in NIKUD:
            out += '0'
        else:
            j += 1
            if j < len(text_non):
                if text_non[j] == c:
                    out += '0'
                else:
                    out += '1' if text_non[j] == YUD else ('2' if text_non[j] == VAV else '?')
                    while j < len(text_non) and text_non[j] != c:
                        if not text_non[j] in YV:
                            return '?'
                        j += 1
                        if not j < len(text_non):
                            return '?'
    return out

class MaleHaserDataset(Dataset):

    def __init__(self, fn='./data/processed/male_haser.csv', tokenizer=None):
        self.df = pd.read_csv(fn)

        self.df['labels'] = self.df.apply(lambda row: make_male_labels(row['nikud'], row['male']), axis=1)
        self.df = self.df[~self.df.labels.str.contains('?', regex=False)].copy()

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        return row.nikud, row.labels

def labels2tensor(labels, maxlen):
    # labels: list or Series
    padded = [
        '0' + w + '0' * (maxlen - len(w) - 1)
        for w in labels
    ]
    return torch.tensor([
        [int(c) for c in p]
        for p in padded
    ])

class MaleHaserCollator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def collate(self, batch):
        X = self.tokenizer(
            [N for N, L in batch],
            padding='longest',
            truncation=True,
            return_tensors='pt'
        )
        y = labels2tensor([L for N, L in batch], X.input_ids.size(1))
        
        return {**X, 'labels': y}