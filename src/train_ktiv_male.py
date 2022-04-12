from dataclasses import dataclass, field
from typing import Optional
import argparse
from datasets import KtivMaleDataset, KtivMaleCollator
from models import KtivMaleModel
from transformers import CanineTokenizer, TrainingArguments, Trainer
import torch

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output_dir', default='models/ktiv_male', help='Save directory for model')
    parser.add_argument('--num_train_epochs', default=3, help='Number of train epochs')
    parser.add_argument('--per_device_train_batch_size', default=32, help='Train batch size')
    parser.add_argument('--per_device_eval_batch_size', default=32, help='Validation batch size')
    parser.add_argument('--save_strategy', default='no', help='Whether to save on every epoch')
    parser.add_argument('--learning_rate', default=5e-5, help='Learning rate')
    parser.add_argument('--adam_beta1', default=0.9, help='AdamW beta1 hyperparameter')
    parser.add_argument('--adam_beta2', default=0.999, help='AdamW beta2 hyperparameter')
    parser.add_argument('--weight_decay', default=0.0, help='Weight decay')
    parser.add_argument('--evaluation_strategy', default='steps', help='How to validate (set to "no" for no validation)')
    parser.add_argument('--eval_steps', default=500, help='Validate every N steps')
    return parser.parse_args()


def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device detected:', device)

    args = parse_arguments()
    training_args = TrainingArguments(**vars(args)) # vars: Namespace to dict

    print('Loading data...')
    train_dataset = KtivMaleDataset(split='train')
    eval_dataset = KtivMaleDataset(split='val')

    print('Loading tokenizer...')
    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    collator = KtivMaleCollator(tokenizer)

    print('Loading model...')
    model = KtivMaleModel.from_pretrained("google/canine-c", num_labels=3)

    print('Creating trainer...')
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator.collate,
    )

    print(f'Training... (on device: {device})')
    trainer.train()

    print('Done')

if __name__ == '__main__':
    main()