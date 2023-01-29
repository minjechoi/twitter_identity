"""A file that contains the various scripts for merging different files
"""

import os
from os.path import join
import gzip
import re

import pandas as pd
import ujson as json
from tqdm import tqdm

from twitter_identity.utils.utils import write_data_file_info, strip_tweet

def merge_splitted_extracted_identities(load_dir,save_dir):
    """Merges the extracted identity shards

    Args:
        load_dir (_type_): _description_
        save_dir (_type_): _description_
    """
    
    for n_changes in ['0_changes','1plus_changes']:
        uid2profiles={}
        valid_files = [file for file in sorted(os.listdir(load_dir)) if n_changes in file]
        for file in tqdm(valid_files):
            identity = 'socialmedia' if 'social_media' in file else file.split('_')[-1]
            with open(join(load_dir,file),'r') as f:
                for line in f:
                    line=line.split('\t')
                    uid,ts,desc=line[:3]
                    desc=desc.strip()
                    if uid not in uid2profiles:
                        uid2profiles[uid] = {}
                    if ts not in uid2profiles[uid]:
                        uid2profiles[uid][ts]=[]
                    if desc:
                        uid2profiles[uid][ts].append(desc)
        
        # save
        print(f"Saving results for {n_changes}")
        with gzip.open(join(save_dir,f'description_changes.{n_changes}.all_identities.json.gz'),'wt') as outf:
            for uid,D1 in tqdm(uid2profiles.items()):
                times=sorted(D1.keys())
                for ts in times:
                    V = uid2profiles[uid][ts]
                    out_line = f'{uid}\t{ts}\t%s\n'%('\t'.join(V))
                    outf.write(out_line)
        # write_data_file_info(__file__, merge_splitted_extracted_identities.__name__, save_dir, [load_dir])
                    
    return

def merge_training_sets(user_id_file, load_dir, save_dir):
    """Loads all tweet files, and sorts them by each user, then by identity category

    Args:
        user_id_file (_type_): _description_
        load_dir (_type_): _description_
        save_dir (_type_): _description_
    """
    
    def collect_training_tweets(tweet_type, uids):
        """iterates through all training samples to collect identity- and tweet_type-relevant tweets

        Args:
            tweet_type (_type_): _description_
            uids (_type_): _description_

        Returns:
            dictionary: contains the tweets for each user
        """
        uid2text = {}
        for file in tqdm(sorted(os.listdir(load_dir))):
            with gzip.open(join(load_dir,file),'rt') as f:
                for line in f:
                    obj=json.loads(line)
                    if tweet_type=='retweets_quotes':
                        # retweets & quotes: get the original tweet that the user ended up retweeting/quoting
                        if obj['tweet_type'] in ['retweet','quote']:
                            uid = obj['user_id']
                            if uid in uids:
                                if obj['lang_origin'] in ['en','und']:
                                    text = obj['text'] if obj['tweet_type']=='retweet' else obj['text_origin'] # differs if it is a retweet or quote
                                    text = strip_tweet(text,url='replace')
                                    if uid not in uid2text:
                                        uid2text[uid]=[]
                                    uid2text[uid].append(text)
                    elif tweet_type=='tweets_replies':
                        # first case: plain tweet, reply, mention, or quote response
                        if obj['tweet_type']!='retweet':
                            uid = obj['user_id']
                            if uid in uids:
                                if obj['lang'] in ['en','und']:
                                    text = obj['text']
                                    text = strip_tweet(text,url='replace')
                                    if uid not in uid2text:
                                        uid2text[uid]=[]
                        # second case: original tweet of retweet or quote
                        if obj['tweet_type'] in ['retweet','quote']:
                            uid = obj['user_id_origin']
                            if uid in uids:
                                if obj['lang_origin'] in ['en','und']:
                                    text = obj['text'] if obj['tweet_type']=='retweet' else obj['text_origin'] # differs if it is a retweet or quote
                                    text = strip_tweet(text,url='replace')
                                    if uid not in uid2text:
                                        uid2text[uid]=[]
                                    uid2text[uid].append(text)

        for uid,texts in uid2text.items():
            uid2text[uid]=list(set(texts))

        return uid2text
    
    df = pd.read_csv(user_id_file,sep='\t',dtype={'user_id':str})
    identities = sorted(df.identity.unique())
    tweet_types = ['retweets_quotes','tweets_replies']
    
    for tweet_type in tweet_types:
        for identity in identities:
            df2 = df[df.identity==identity]
            uid2label = {uid:label for uid,label in df2.values}
            uids = set(df2['user_id'])
            # get all relevant tweets
            print(f'Loading all files and collecting tweets for {tweet_type} / {identity}')
            uid2text = collect_training_tweets(tweet_type, uids)
            
            # save to file
            print(f'Writing tsv file for {tweet_type} / {identity}')
            with gzip.open(join(save_dir,f'{tweet_type}.{identity}.tsv.gz'),'wt') as f:
                for uid,texts in tqdm(uid2text.items()):
                    label = uid2label[uid]
                    f.write(f'{uid}\t{label}\t'+'\t'.join(texts[:100])+'\n')
    write_data_file_info(__file__,merge_training_sets.__name__,save_dir,[load_dir])
    return

if __name__=='__main__':
    # merge the identity files
    # load_dir = '/shared/3/projects/bio-change/data/interim/description_changes/extracted/splitted'
    # save_dir = '/shared/3/projects/bio-change/data/interim/description_changes/extracted/'
    # merge_splitted_extracted_identities(load_dir, save_dir)
    
    # create training data out of the collected tweets
    user_id_file = '/shared/3/projects/bio-change/data/interim/identity_classifier-train_data/positive-negative-users/labels_200000.df.tsv'
    load_dir = '/shared/3/projects/bio-change/data/raw/identity_classifier-train_data'
    save_dir = '/shared/3/projects/bio-change/data/interim/identity_classifier-train_data/positive-negative-tweets'
    merge_training_sets(user_id_file, load_dir, save_dir)