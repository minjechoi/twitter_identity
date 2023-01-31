"""
Code related to the model
"""
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.nn.functional import cross_entropy
from transformers import (
    AutoModelForSequenceClassification, 
    get_linear_schedule_with_warmup,
    get_constant_schedule
)
from pytorch_lightning import LightningModule
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
import numpy as np

class IdentityClassifier(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        model_cache_dir:str,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: float = 0.06,
        weight_decay: float = 0.01,
        dropout_prob: float=0.1,
        weighted_class_loss: bool=False,
        **kwargs,

    ):
        super().__init__()
        self.save_hyperparameters()

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("IdentityClassifier")
        parser.add_argument("--model_name_or_path", type=str, default=None)
        parser.add_argument("--model_cache_dir", type=str,
                            default='/shared/3/projects/bio-change/.cache/')
        parser.add_argument("--learning_rate", type=float, default=1e-5)
        parser.add_argument("--adam_epsilon", type=float, default=1e-8)
        parser.add_argument("--warmup_steps", type=float, default=0.0)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--dropout_prob", type=float, default=0.1)
        parser.add_argument("--weighted_class_loss", action='store_true')
        return parent_parser

    def setup(self, stage: str = None) -> None:
        # load model
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(
            self.hparams.model_name_or_path,
            cache_dir=self.hparams.model_cache_dir,
            num_labels=2,
            ignore_mismatched_sizes=True
            )
        
        # self.bert_model = torch.compile(self.bert_model)
        
        if stage!='fit':
            return
        
        # load class weights
        self.class_weights = torch.tensor([1,self.trainer.datamodule.datasets['train'].weight]) if self.hparams.weighted_class_loss else None
        
        # Calculate total steps
        train_loader = self.trainer.datamodule.train_dataloader()
        tb_size = self.trainer.datamodule.hparams.train_batch_size
        if self.trainer.max_epochs>0:
            self.total_steps= len(train_loader) * self.trainer.max_epochs
        else:
            self.total_steps = self.trainer.max_steps

        if self.hparams.warmup_steps<=1:
            self.hparams.warmup_steps = int(self.total_steps*self.hparams.warmup_steps)
        else:
            self.hparams.warmup_steps = int(self.hparams.warmup_steps)

        print('Max steps:',self.total_steps)
        print("tb size",tb_size)
        print("Len train loader",len(train_loader))
        print("Max epochs",self.trainer.max_epochs)
        print('Warmup steps:',self.hparams.warmup_steps)
        return

    def forward(self, batch):
        outputs = self.bert_model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['token_type_ids'] if 'token_type_ids' in batch else None,
            )
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        logits = outputs.logits
        labels = batch['labels']
        if self.hparams.weighted_class_loss:
            loss=cross_entropy(logits, labels, weight=self.class_weights.to(labels.device))
        else:
            loss=cross_entropy(logits, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        result = {}

        outputs = self.forward(batch)
        logits = outputs.logits
        labels = batch['labels']
        if self.hparams.weighted_class_loss:
            loss=cross_entropy(logits, labels, weight=self.class_weights.to(labels.device))
        else:
            loss=cross_entropy(logits, labels)
        y_pred = logits.argmax(1).cpu().detach().tolist()
        y_score = logits.softmax(1)[:,1].cpu().detach().tolist()
        y_true = labels.cpu().detach().tolist()
        loss = loss.item()

        result['answers'] = y_true
        result['preds'] = y_pred
        result['scores'] = y_score
        result['loss'] = loss
        return result

    def validation_epoch_end(self, outputs):
        log_dict={}
        all_preds = []
        all_answers = []
        all_scores = []
        all_losses = []
        for output in outputs:
            all_answers.extend(output['answers'])
            all_preds.extend(output['preds'])
            all_scores.extend(output['scores'])
            all_losses.append(output['loss'])
            
        acc = accuracy_score(y_true=all_answers,y_pred=all_preds)
        f1 = f1_score(y_true=all_answers,y_pred=all_preds)
        if len(set(all_answers))>1:
            auc = roc_auc_score(y_true=all_answers,y_score=all_scores)
        else:
            auc = 0
        log_dict['val_acc'] = round(acc,3)
        log_dict['val_f1'] = round(f1,3)
        log_dict['val_auc'] = round(auc,3)
        log_dict['val_loss'] = round(np.mean(all_losses),3)
        self.log_dict(log_dict, prog_bar=True)
        return
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def test_epoch_end(self, outputs):
        log_dict={}
        all_preds = []
        all_answers = []
        all_scores = []
        all_losses = []
        for output in outputs:
            all_answers.extend(output['answers'])
            all_preds.extend(output['preds'])
            all_scores.extend(output['scores'])
            all_losses.append(output['loss'])
            
        acc = accuracy_score(y_true=all_answers,y_pred=all_preds)
        f1 = f1_score(y_true=all_answers,y_pred=all_preds)
        if len(set(all_answers))>1:
            auc = roc_auc_score(y_true=all_answers,y_score=all_scores)
        else:
            auc = 0
        log_dict['test_acc'] = round(acc,3)
        log_dict['test_f1'] = round(f1,3)
        log_dict['test_auc'] = round(auc,3)
        log_dict['test_loss'] = round(np.mean(all_losses),3)
        self.log_dict(log_dict,prog_bar=False)
        return log_dict

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.bert_model.named_parameters() if not any(nd in n for nd in no_decay)]+\
                [p for n, p in self.bert_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.bert_model.named_parameters() if any(nd in n for nd in no_decay)]+\
                    [p for n, p in self.bert_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, 
            lr=self.hparams.learning_rate, 
            eps=self.hparams.adam_epsilon,
            )

        if self.hparams.warmup_steps>0:
            scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.total_steps
            )
        else:
            scheduler = get_constant_schedule(
                optimizer,
            )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]