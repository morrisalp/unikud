from hebrew_utils import NIKUD, YUD, VAV, ABG, N_VOWELS, idx2chr
from tqdm.auto import tqdm
import numpy as np
import torch

class KtivMaleTask:

    def __init__(self, tokenizer, model, device='cpu'):
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.device = device

    def _nikud2male_word(self, word, logits, sample=False, sample_thresh=0.1):
        
        if sample:
            probs = logits.softmax(axis=-1).cpu().numpy()
            
            # remove probabilities under sample_thresh and normalize
            probs = np.where(probs < 0.1, 0, probs)
            probs /= probs.sum(axis=-1)[:, None]
            
            output = ''
            for c, P in zip(word, probs[1:]):
                if c not in NIKUD:
                    output += np.random.choice(['', YUD, VAV], p=P)
                    output += c

            return output
            
        else:
            preds = logits.argmax(axis=-1).cpu().numpy()
            output = ''
            for c, L in zip(word, preds[1:]):
                if L == 1:
                    output += YUD
                if L == 2:
                    output += VAV
                if c not in NIKUD:
                    output += c

            return output

    def _nikud2male_batch(self, batch, **kwargs):
        
        # if all words in batch are too small, model cannot process them so just return unchanged
        if all(len(word) <= 1 for word in batch):
            for word in batch:
                yield ''.join([c for c in word if c not in NIKUD])
        
        else:
            X = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(self.device)
            logits = self.model(**X).logits.detach()

            for i, word in enumerate(batch):
                yield self._nikud2male_word(word, logits[i], **kwargs)

    def nikud2male(self, text, split=False, pbar=False, sample=False, sample_thresh=0.1, batch_size=64):
        """
        text: Hebrew text with nikud
        returns: Hebrew text in ktiv male without nikud
        """
        words = text.split(' ') if split else [text]
        batches = [[]]
        for word in words:
            if len(batches[-1]) < batch_size:
                batches[-1] += [word]
            else:
                batches += [[word]]
        
        outputs = [
            out
            for batch in (tqdm(batches) if pbar else batches)
            for out in self._nikud2male_batch(batch, sample=sample, sample_thresh=sample_thresh)
        ]
        return ' '.join(outputs)


class NikudTask:

    def __init__(self, tokenizer, model, device='cpu', max_len=2046):
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.device = device
        self.max_len = max_len
        # Note: max_len is 2 less than model's input length 2048,
        # to account for BOS and EOS tokens
        
    def _decode_nikud_probs(self, probs, d_thresh=0.5, v_thresh=0.5, o_thresh=0.5):
        # probs: N_TARGET_LABELS probabilities for nikkud for a single character, or deletion (last prob)

        # Note: first N_VOWELS are mutually exclusive vowels
        # next are dagesh, shin dot, and sin dot
        # finally the deletion flag
        
        vowel_probs = probs[:N_VOWELS]
        other_probs = probs[N_VOWELS:-1]
        del_prob = probs[-1]
        
        maxvow = vowel_probs.max().item()
        argmaxvow = vowel_probs.argmax().item()
        
        if del_prob > d_thresh:
            return None # special symbol for deletion
    
        out = ''
    
        if maxvow > v_thresh:
            out += idx2chr[argmaxvow]

        for i, p in enumerate(other_probs):
            if p > o_thresh:
                out += idx2chr[N_VOWELS + i]

        return out
        
    
    def add_nikud(self, text, **kwargs):
        
        assert len(text) <= self.max_len, f'Input text cannot be longer than {self.max_len} characters.'
        
        X = self.tokenizer([text], return_tensors='pt').to(self.device)
        logits = self.model(**X).logits.detach()[0]
        probs = torch.sigmoid(logits)
        
        output = ''
        for i, char in enumerate(text):
            output += char
            if char in ABG:
                char_probs = probs[i + 1]

                decoded = self._decode_nikud_probs(char_probs, **kwargs)

                if decoded is None and len(output) > 0:
                    output = output[:-1]
                else:
                    output += decoded

        return output


if __name__ == '__main__':
    from models import KtivMaleModel, UnikudModel
    from transformers import CanineTokenizer

    print('Loading tokenizer')
    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    print('Loading KM model')
    model = KtivMaleModel.from_pretrained("google/canine-c", num_labels=3)

    print('Loading KM task')
    km_task = KtivMaleTask(tokenizer, model)
    print('KM task loaded')
    text = 'אָבִיב הוֹלֵךְ וּבָא אִתּוֹ רַק אֹשֶׁר וְשִׂמְחָה'
    print(text)
    print(km_task.nikud2male(text, split=True, pbar=True))
    
    print('Loading UNIKUD model')
    model = UnikudModel.from_pretrained("google/canine-c", num_labels=3)
    
    print('Loading nikud task')
    n_task = NikudTask(tokenizer, model)
    print('Nikud task loaded')
    text = 'זאת דוגמא של טקסט לא מנוקד בעברית'
    print(text)
    print(n_task.add_nikud(text))