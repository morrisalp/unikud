from dataclasses import dataclass, field
from typing import Optional
import argparse
from datasets import NikudDataset, NikudCollator
from models import UnikudModel
from metrics import unikud_metrics
from transformers import CanineTokenizer, TrainingArguments, Trainer
import torch

OUTPUT_DIR = 'models/unikud/latest'

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Save directory for model')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='Number of train epochs')
    parser.add_argument('--per_device_train_batch_size', type=int, default=2, help='Train batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps (train)')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=2, help='Validation batch size')
    parser.add_argument('--save_strategy', type=str, default='no', help='Whether to save on every epoch ("epoch"/"no")')
    parser.add_argument('--learning_rate',  type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--lr_scheduler_type',  type=str, default='linear', help='Learning rate scheduler type ("linear"/"cosine"/"constant"/...')
    parser.add_argument('--warmup_ratio',  type=float, default=0.0, help='Warmup ratio')
    parser.add_argument('--adam_beta1', type=float, default=0.9, help='AdamW beta1 hyperparameter')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help='AdamW beta2 hyperparameter')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--evaluation_strategy', type=str, default='steps', help='How to validate (set to "no" for no validation)')
    parser.add_argument('--eval_steps', type=int, default=10000, help='Validate every N steps')
    return parser.parse_args()


def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device detected:', device)

    args = parse_arguments()
    training_args = TrainingArguments(**vars(args)) # vars: Namespace to dict

    print('Loading data...')
    train_dataset = NikudDataset(split='train')
    eval_dataset = NikudDataset(split='val')

    print('Loading tokenizer...')
    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    collator = NikudCollator(tokenizer)

    print('Loading model...')
    model = UnikudModel.from_pretrained("google/canine-c")

    print('Creating trainer...')
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator.collate,
        compute_metrics=unikud_metrics
    )

    print(f'Training... (on device: {device})')
    trainer.train()
    
    print(f'Saving to: {OUTPUT_DIR}')
    trainer.save_model(f'{OUTPUT_DIR}')

    print('Done')

if __name__ == '__main__':
    main()
