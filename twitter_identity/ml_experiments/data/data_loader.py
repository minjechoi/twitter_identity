import gzip
import re

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer
)
import pandas as pd
from tqdm import tqdm
from pytorch_lightning import LightningDataModule

class TextDataset(Dataset):
    def __init__(self, data_file, col_name='text'):
        data = []
        df=pd.read_csv(data_file,sep='\t')
        for label,text in df[['label',col_name]].values:
            text=re.sub(r'(URL|@username)','',text).strip()
            text = text.strip()
            data.append((text,int(label)))
            
        # with gzip.open(data_file,'rt') as f:
        #     for line in f:
        #         uid,label,text=line.split('\t')
        #         text=re.sub(r'(URL|@username)','',text).strip()
        #         text = text.strip()
        #         data.append((text,int(label)))
    
        self.data = data
        
        # compute class weights
        n_pos = sum([x[1] for x in data])
        self.weight = (len(data)-n_pos)/n_pos
        return

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class PredictTextDataset(Dataset):
    def __init__(self, data_file, col_name='text'):
        data = []
        df = pd.read_csv(data_file,sep='\t',dtype={'user_id':str})
        df[[col_name]] = df[[col_name]].fillna('None')
        self.data = df[col_name].values
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
                predict_file: str,
                col_name: str='text',
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
        parser.add_argument("--predict_file", type=str)
        parser.add_argument("--col_name", type=str, default='text')
        parser.add_argument("--train_batch_size", type=int, default=8)
        parser.add_argument("--eval_batch_size", type=int, default=16)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--max_length", type=int, default=512)
        return parent_parser

    # sets up train, test & val datasets
    def setup(self, stage: str):
        # stages are always one of 'fit', 'validate', 'test' or 'predict'
        # set datasets
        self.datasets = {}
        
        if stage=='predict':
            self.datasets['predict'] = PredictTextDataset(
                data_file = self.hparams.predict_file,
                col_name=self.hparams.col_name)
            # self.datasets['predict'] = TextDataset(
            #     data_file = self.hparams.predict_file,
            #     col_name=self.hparams.col_name)
        
        else:
            if stage=='fit':
                self.datasets = {
                    'train': TextDataset(data_file=self.hparams.train_file),
                    'val': TextDataset(data_file=self.hparams.val_file),
                    'test': TextDataset(data_file=self.hparams.test_file),            
                }        
                # save class weights
                self.weight = self.datasets['train'].weight
            elif stage=='test':
                self.datasets = {
                    'test': TextDataset(data_file=self.hparams.test_file),            
                }
            elif stage=='predict':
                self.datasets = {
                    'predict': TextDataset(data_file=self.hparams.predict_file),            
                }
        
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.model_name_or_path,
            cache_dir=self.hparams.model_cache_dir
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

    def predict_collate_fn(self,texts):
        
        # texts2 = []
        # for text in texts:
        #     if len(text.strip())>0:
        #         texts2.append(text)
        #     else:
        #         texts2.append('None')
        # aggregate batch data
        # max_ln=0
        # print(texts)
        # print('\n')
        # print(texts[0])
        # print(type(texts))
        
        outputs = self.tokenizer.batch_encode_plus(
            texts,
            max_length=self.hparams.max_length, 
            padding='longest',
            truncation='longest_first', 
            return_length=False
            )
        for k,v in outputs.items():
            outputs[k]=torch.tensor(v)
        return outputs

    def train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.hparams.train_batch_size, shuffle=True, num_workers=self.hparams.num_workers, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.datasets["val"], batch_size=self.hparams.eval_batch_size, shuffle=False, num_workers=self.hparams.num_workers, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.hparams.eval_batch_size, shuffle=False, num_workers=self.hparams.num_workers, collate_fn=self.collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.datasets["predict"], batch_size=self.hparams.eval_batch_size, shuffle=False, num_workers=self.hparams.num_workers, collate_fn=self.predict_collate_fn)
