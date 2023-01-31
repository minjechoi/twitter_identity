import argparse
import sys
import os
from os.path import join
import json

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import loggers as pl_loggers

from models.classifier import *
from data.data_loader import *

# torch.set_float32_matmul_precision('medium')

def train(args):
    dict_args = vars(args)
    seed_everything(args.seed)
    
    # load logger
    print(os.getpid())
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.default_root_dir)
        
    # modules for checkpointing
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, 
        monitor="val_f1",
        mode="max",
        dirpath=args.default_root_dir,
        filename="checkpoint-{epoch}-{val_f1:.2f}-{val_auc:.2f}-{val_loss:.2f}",
        every_n_epochs=1, 
        save_weights_only=True)

    early_stop_callback = EarlyStopping(
        monitor="val_f1",
        mode="max",
        min_delta=0.0, 
        patience=args.patience, 
        verbose=False,
    )
    
    # load datamodule and model
    dm = DataModuleForIdentityClassification(**dict_args)
    model = IdentityClassifier(**dict_args)

    # Trainer w/ logger and checkpointing
    trainer = Trainer.from_argparse_args(
        args, 
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stop_callback])

    # Fit model
    trainer.fit(model, dm) # now including checkpoint for training

    # Test best model and save results
    result = trainer.test(
        model,
        datamodule=dm, 
        ckpt_path='best')[0]
    
    with open(join(args.default_root_dir, 'results.json'), 'w') as f:
        f.write(json.dumps(result))

    # remove checkpoint if not saving
    if args.remove_checkpoint:
        for file in os.listdir(args.default_root_dir):
            if file.endswith('ckpt'):
                os.remove(os.path.join(args.default_root_dir,file))    
    return

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience",type=int, default=0, help="Number consecutive epochs to tolerate if validation metric does not improve")
    parser.add_argument("--remove_checkpoint", action='store_true', help='Enable if not willing to save checkpoints, which might be large')
    parser.add_argument('--tweet_type', type=str, choices=['tweet','retweet'], default=None)
    parser.add_argument('--identity', type=str, default=None)
    parser = Trainer.add_argparse_args(parser)
    parser = IdentityClassifier.add_model_specific_args(parser)
    parser = DataModuleForIdentityClassification.add_model_specific_args(parser)
    args = parser.parse_args()
    
    if (args.tweet_type is not None) and (args.identity is not None):
        if args.tweet_type=='tweet':
            tweet_type='tweets_replies'
        elif args.tweet_type=='retweet':
            tweet_type='retweets_quotes'
        identity=args.identity            
        args.train_file = f'/shared/3/projects/bio-change/data/processed/identity_classifier-train_data/{tweet_type}.{identity}.train.tsv.gz'
        args.val_file = args.train_file.replace('.train.tsv.gz','.val.tsv.gz')
        args.test_file = args.train_file.replace('.train.tsv.gz','.test.tsv.gz')
        args.default_root_dir = f'/shared/3/projects/bio-change/results/experiments/identity-classifier/{tweet_type}/{identity}'
        args.devices=1
        args.precision=16
        args.accelerator='gpu'
        args.model_name_or_path='cardiffnlp/tweet-topic-21-multi'
        args.max_epochs=10
        args.warmup_steps=0.06
        args.weighted_class_loss=False
        args.patience=1
    train(args)
            
