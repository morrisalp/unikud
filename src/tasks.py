from hebrew_utils import NIKUD, YUD, VAV
from tqdm.auto import tqdm
import numpy as np

class KtivMaleTask:

    def __init__(self, tokenizer, model, device='cpu'):
        self.tokenizer = tokenizer
        self.model = model.to(device)

        self.device = device

    def _nikud2male_word(self, word, logits, sample=False, sample_thresh=0.1):
        
        if len(word) < 2:
            return ''.join([c for c in word if c not in NIKUD])
            # ^ Needed because model throws error when input is too small
        
        X = self.tokenizer(word, return_tensors='pt')
        X = X.to(self.device)
        logits = self.model(**X).logits.detach()
        
        if sample:
            probs = logits.softmax(axis=-1)[0].cpu().numpy()
            
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
            preds = logits.argmax(-1)[0].cpu().numpy()
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


# class NikudTask:

#     def __init__(self, tokenizer, model):
#         self.tokenizer = tokenizer
#         self.model = model

if __name__ == '__main__':
    from models import KtivMaleModel
    from transformers import CanineTokenizer

    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    model = KtivMaleModel.from_pretrained("google/canine-c", num_labels=3)

    print('Loading task')
    km_task = KtivMaleTask(tokenizer, model)
    print('Task loaded')
    text = 'אָבִיב הוֹלֵךְ וּבָא אִתּוֹ רַק אֹשֶׁר וְשִׂמְחָה'
    print(text)
    print(km_task.nikud2male(text, split=True, pbar=True))