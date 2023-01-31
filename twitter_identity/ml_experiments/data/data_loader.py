import os
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer
)
import pandas as pd
from tqdm import tqdm
from pytorch_lightning import LightningDataModule

class TextDataset(Dataset):
    def __init__(self, data_file):
        data = []
        with open(data_file) as f:
            for line in f:
                uid,label,text=line.split('\t')
                text = text.strip()
                data.append((text,int(label)))
        
        self.data = data
        return

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class DataModuleForIdentityClassification(LightningDataModule):

    # Init function - only sets hyperparameters and tokenizer
    def __init__(self,
                model_name_or_path: str,
                model_cache_dir: str,
                train_file: str,
                test_file: str,
                val_file: str,
                max_length: int = 512,
                train_batch_size: int = 8,
                eval_batch_size: int = 16,
                num_workers: int = 4,
                **kwargs, 
                ):
        super().__init__()
        self.save_hyperparameters()
        return

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DataModuleForIdentityClassification")
        parser.add_argument("--train_file", type=str)
        parser.add_argument("--val_file", type=str)
        parser.add_argument("--test_file", type=str)
        parser.add_argument("--train_batch_size", type=int, default=8)
        parser.add_argument("--eval_batch_size", type=int, default=16)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--max_length", type=int, default=512)
        return parent_parser

    # sets up train, test & val datasets
    def setup(self, stage: str):
        # set datasets
        self.datasets = {
            'train': TextDataset(data_file=self.hparams.train_file),
            'val': TextDataset(data_file=self.hparams.val_file),
            'test': TextDataset(data_file=self.hparams.test_file),            
        }
        
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.model_name_or_path,
            cache_dir=self.model_cache_dir
        )
        return

    def collate_fn(self,data):
        # aggregate batch data
        max_ln=0
        texts, labels = [],[]
        for text,label in data:
            texts.append(text)
            labels.append(label)
            
        outputs = self.tokenizer.batch_encode_plus(
            texts,
            max_length=self.hparams.max_length, 
            padding='longest',
            truncation='longest_first', 
            return_length=False
            )
        for k,v in outputs.items():
            outputs[k]=torch.tensor(v)
        outputs['labels'] = torch.LongTensor(labels)
        return outputs

    def train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.datasets["val"], batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn)
