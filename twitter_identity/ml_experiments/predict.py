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

def predict(args):
    dict_args = vars(args)
        
    # load datamodule and model
    dm = DataModuleForIdentityClassification(**dict_args)
    model = IdentityClassifier(**dict_args)

    # Trainer w/ logger and checkpointing
    trainer = Trainer.from_argparse_args(args)

    # Test best model and save results
    results = trainer.predict(
        model, 
        datamodule=dm, 
        ckpt_path=args.ckpt_path)
    
    results2 = []
    for arr in results:
        results2.extend(arr)
        
    with open(join(args.default_root_dir,args.save_file), 'w') as f:
        for val in results2:
            val=round(val,3)
            f.write(f'{val}\n')
    print(f'Saved in {args.default_root_dir}/{args.save_file}')
    return

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--identity', type=str, default=None)
    parser.add_argument('--activity_type', type=str) # 
    parser.add_argument('--tweet_type', type=str)
    parser.add_argument('--save_file', type=str, default=None)
    

    parser = Trainer.add_argparse_args(parser)
    parser = IdentityClassifier.add_model_specific_args(parser)
    parser = DataModuleForIdentityClassification.add_model_specific_args(parser)
    args = parser.parse_args()
    
    data_dir='/shared/3/projects/bio-change/data/interim/activities_by_treated_users/tweets'
    all_identities = sorted(set([x.split('.')[1] for x in sorted(os.listdir(data_dir))]))
    for identity in all_identities:
        args.identity=identity
        args.predict_file = f'/shared/3/projects/bio-change/data/interim/activities_by_treated_users/tweets/{args.activity_type}.{args.identity}.df.tsv.gz'
        args.default_root_dir = '/shared/3/projects/bio-change/data/interim/activities_by_treated_users/predicted_scores/'
        # if args.save_file==None:
        args.save_file = f'{args.tweet_type}-classifier.{args.identity}.{args.activity_type}.txt'
        
        # get checkpoint path
        if args.tweet_type=='tweet':
            model_dir = f'/shared/3/projects/bio-change/results/experiments/identity-classifier/tweets_replies/{args.identity}'
        elif args.tweet_type=='retweet':
            model_dir = f'/shared/3/projects/bio-change/results/experiments/identity-classifier/retweets_quotes/{args.identity}'
        try:
            model_file = [x for x in os.listdir(model_dir) if x.endswith('.ckpt')][0]
        except:
            continue
        args.ckpt_path = join(model_dir,model_file)
        
        args.devices=1
        args.precision=16
        args.accelerator='gpu'
        args.model_name_or_path='cardiffnlp/tweet-topic-21-multi'
        
        
        # get all identities
        predict(args)
