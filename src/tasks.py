from hebrew_utils import NIKUD, YUD, VAV
from tqdm.auto import tqdm

class KtivMaleTask:

    def __init__(self, tokenizer, model, device='cpu'):
        self.tokenizer = tokenizer
        self.model = model.to(device)

        self.device = device

    def _nikud2male_word(self, word):
        
        if len(word) < 2:
            return ''.join([c for c in word if c not in NIKUD])
            # ^ Needed because model throws error when input is too small
        
        X = self.tokenizer(word, return_tensors='pt')
        X = X.to(self.device)
        preds = self.model(**X).logits.argmax(-1)[0].cpu().numpy()
        
        output = ''
        
        for c, L in zip(word, preds[1:]):
            if L == 1:
                output += YUD
            if L == 2:
                output += VAV
            if c not in NIKUD:
                output += c
        
        return output

    def nikud2male(self, text, split=False, pbar=False):
        """
        text: Hebrew text with nikud
        returns: Hebrew text in ktiv male without nikud
        """
        words = text.split(' ') if split else [text]
        outputs = [
            self._nikud2male_word(word)
            for word in (tqdm(words) if pbar else words)
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