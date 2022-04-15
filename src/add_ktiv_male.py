from models import KtivMaleModel
from tasks import KtivMaleTask
from transformers import CanineTokenizer
import pandas as pd
import torch
from tqdm.auto import tqdm

MODEL_FN =  'models/ktiv_male/latest'
DATA_FN = 'data/processed/nikud.csv'
SAVE_FN = 'data/processed/nikud_with_ktiv_male.csv'

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device detected:', device)

    print('Loading data...')
    df = pd.read_csv(DATA_FN)

    print('Loading tokenizer...')
    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")

    print('Loading model...')
    model = KtivMaleModel.from_pretrained(MODEL_FN)
    model.to(device)
    model.eval()

    print('Creating task...')
    task = KtivMaleTask(tokenizer, model, device=device)

    print('Adding ktiv male to text...')
    tqdm.pandas(desc='Generating ktiv male')
    df.text = df.text.progress_apply(lambda text: task.nikud2male(text, split=True))

    
    print(f'Saving to: {SAVE_FN}')
    df.to_csv(SAVE_FN, index=False)


if __name__ == '__main__':
    main()