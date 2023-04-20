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

def eval(args):
    dict_args = vars(args)
        
    # load datamodule and model
    dm = DataModuleForIdentityClassification(**dict_args)
    model = IdentityClassifier(**dict_args)

    # Trainer w/ logger and checkpointing
    trainer = Trainer.from_argparse_args(args)

    # Test best model and save results
    results = trainer.test(
        model, 
        datamodule=dm, 
        ckpt_path=args.ckpt_path)[0]
    
    with open(join(args.default_root_dir, 'results.json'), 'w') as f:
        f.write(json.dumps(results))

    return

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default=None)
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
        
        # get checkpoint path
        if args.tweet_type=='tweet':
            model_dir = f'/shared/3/projects/bio-change/results/experiments/identity-classifier/tweets_replies/{args.identity}'
        elif args.tweet_type=='retweet':
            model_dir = f'/shared/3/projects/bio-change/results/experiments/identity-classifier/retweets_quotes/{args.identity}'
        model_file = [x for x in os.listdir(model_dir) if x.endswith('.ckpt')][0]
        args.ckpt_path = join(model_dir,model_file)
        
    eval(args)
