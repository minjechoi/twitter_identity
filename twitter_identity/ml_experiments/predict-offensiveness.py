import argparse
import os
from os.path import join

import pytorch_lightning as pl
from pytorch_lightning import Trainer

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
        datamodule=dm
        )
    
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
    
    # args.tweet_type = 'tweet'
    # args.activity_type = 'activities_origin'
    
    # get checkpoint path    
    args.devices=1
    args.precision=16
    args.accelerator='gpu'
    args.model_name_or_path = 'cardiffnlp/twitter-roberta-base-offensive'
    
    identity_dir='/shared/3/projects/bio-change/data/interim/activities_by_treated_users/all_tweets'
    
    if args.identity=='all':
        identities = sorted(set([x.split('.')[1] for x in sorted(os.listdir(identity_dir))]))
    else:
        identities = [args.identity]
            
    if args.activity_type=='all':
        activity_types = ['activities_made','activities_origin']
    else:
        activity_types = [args.activity_type]
    
    if args.tweet_type=='all':
        tweet_types = ['tweet','retweet']
    else:
        tweet_types = [args.tweet_type]
    
    for identity in identities:
        for activity_type in activity_types:
            for tweet_type in tweet_types:
                args.predict_file=f'/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/tweets_by_identity/{activity_type}.{identity}.{tweet_type}.df.tsv.gz'
                args.default_root_dir = '/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/offensiveness-scores'
                args.save_file = f'{activity_type}.{identity}.{tweet_type}.txt'
                    
                # get all identities
                predict(args)

    # else:
    #     args.predict_file=f'/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/identity_added-with_tweet_identity/mentions-from-api/aggregated-interactions/all_original_tweets.df.tsv'
    #     args.default_root_dir = '/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/identity_added-with_tweet_identity/mentions-from-api/aggregated-interactions/'
    #     args.save_file = 'offensive-original.txt'
    #     # args.predict_file=f'/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/identity_added-with_tweet_identity/mentions-from-api/aggregated-interactions/all_replies.df.tsv'
    #     # args.default_root_dir = '/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/identity_added-with_tweet_identity/mentions-from-api/aggregated-interactions/'
    #     # args.save_file = 'offensive-replies.txt'
        
    #     # get all identities
    #     predict(args)
        