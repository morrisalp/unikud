from hebrew_utils import NIKUD, YUD, VAV

class KtivMaleTask:

    def __init__(self, tokenizer, model, device='cpu'):
        self.tokenizer = tokenizer
        self.model = model.to(device)

        self.device = device

    def nikud2male(self, text):
        """
        text: Hebrew text with nikud
        returns: Hebrew text in ktiv male without nikud
        """
        X = self.tokenizer(text, return_tensors='pt')
        X = X.to(self.device)
        preds = self.model(**X).logits.argmax(-1)[0].cpu().numpy()
        
        output = ''
        
        for c, L in zip(text, preds[1:]):
            if L == 1:
                output += YUD
            if L == 2:
                output += VAV
            if c not in NIKUD:
                output += c
        
        return output


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
    print(km_task.nikud2male(text))