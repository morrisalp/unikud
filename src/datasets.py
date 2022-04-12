from random import random
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

class KtivMaleDataset(Dataset):

    def __init__(self, fn='./data/processed/ktiv_male.csv', tokenizer=None, split=None, val_size=0.1):
        self.df = pd.read_csv(fn)
        # random shuffle (with fixed seed) so val split is not biased:
        self.df = self.df.sample(self.df.shape[0], random_state=0)

        if split is not None:
            assert split in ['train', 'val']
            N_TRAIN = int(self.df.shape[0] * (1 - val_size))
            N_VAL = self.df.shape[0] - N_TRAIN
            if split == 'train':
                self.df = self.df.head(N_TRAIN)
            else:
                self.df = self.df.tail(N_VAL)

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

class KtivMaleCollator:

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

if __name__ == '__main__':

    from transformers import CanineTokenizer

    tokenizer = CanineTokenizer.from_pretrained("google/canine-s")

    m = KtivMaleDataset(split='val')
    print('LEN:', len(m))
    c = KtivMaleCollator(tokenizer)

    I = iter(m)

    batch = [next(I), next(I), next(I)]

    out = c.collate(batch)
    print(out)
    for k in out:
        print(k, out[k].shape)