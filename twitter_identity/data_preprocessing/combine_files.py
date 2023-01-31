"""A file that contains the various scripts for merging different files
"""

import os
from os.path import join
import gzip
import re
from multiprocessing import Pool
from random import sample

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
            identity = file.split('_')[-1]
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


def collect_training_tweets(tweet_type, uids, load_dir, aux_load_dir=None):
    """iterates through all training samples to collect identity- and tweet_type-relevant tweets

    Args:
        tweet_type (_type_): _description_
        uids (_type_): _description_

    Returns:
        dictionary: contains the tweets for each user
    """
    uid2text = {}
    # load tweets from load_dir, where the decahose tweets are stored
    for file in sorted(os.listdir(load_dir)):
        with gzip.open(join(load_dir,file),'rt') as f:
            for line in f:
                obj=json.loads(line)
                if tweet_type=='retweets_quotes':
                    # retweets & quotes: get the original tweet that the user ended up retweeting/quoting
                    if obj['tweet_type'] in ['retweet','quote']:
                        uid = obj['user_id']
                        if uid in uids:
                            lang = obj['lang'] if obj['tweet_type']=='retweet' else obj['lang_origin']
                            if lang in ['en','und']:
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
                            lang = obj['lang'] if obj['tweet_type']=='retweet' else obj['lang_origin']
                            if lang in ['en','und']:
                                text = obj['text'] if obj['tweet_type']=='retweet' else obj['text_origin'] # differs if it is a retweet or quote
                                text = strip_tweet(text,url='replace')
                                if uid not in uid2text:
                                    uid2text[uid]=[]
                                uid2text[uid].append(text)
                                
    # load tweets from aux_load_dir, where the api tweets are stored
    if aux_load_dir:
        for file in sorted(os.listdir(aux_load_dir)):
            with open(join(aux_load_dir,file)) as f:
                for line in f:
                    obj=json.loads(line)
                    uid=obj['author_id']
                    if uid in uids:
                        if tweet_type=='tweets_replies':
                            if 'referenced_tweets' not in obj:
                                text=obj['text']
                                text = strip_tweet(text,url='replace')
                                if uid not in uid2text:
                                    uid2text[uid]=[]
                                uid2text[uid].append(text)
                            else:
                                if obj['referenced_tweets'][0]['type']=='replied_to':
                                    text=obj['text']
                                    text = strip_tweet(text,url='replace')
                                    if uid not in uid2text:
                                        uid2text[uid]=[]
                                    uid2text[uid].append(text)
                        
                        elif tweet_type=='retweets_quotes':
                            if 'referenced_tweets' in obj:
                                if obj['referenced_tweets'][0]['type'] in ['retweeted','quoted']:
                                    text=obj['text']
                                    text = strip_tweet(text,url='replace')
                                    if uid not in uid2text:
                                        uid2text[uid]=[]
                                    uid2text[uid].append(text)
                    

    for uid,texts in uid2text.items():
        uid2text[uid]=list(set(texts))

    return uid2text        


def merge_training_sets_worker(tweet_type, identity, user_id_file, load_dir, aux_load_dir, save_dir,n_tweets_per_sample=5,max_samples_per_user=5):
    """Loads a file and gets

    Args:
        tweet_type (_type_): _description_
        identity (_type_): _description_
        user_id_file (_type_): _description_
        load_dir (_type_): _description_
        aux_load_dir (_type_): _description_
    """
    # load set of user ids
    df = pd.read_csv(user_id_file,sep='\t',dtype={'user_id':str})
    df2 = df[df.identity==identity]
    uid2label = {uid:label for uid,_,label in df2.values}
    uids = set(df2['user_id'])
    # get all relevant tweets
    print(f'Loading all files and collecting tweets for {tweet_type} / {identity}')
    
    # collect tweets
    if not identity in ['age_13-17',
        'age_18-24',
        'age_25-34',
        'age_35-49',
        'age_50+',
        'ethnicity_african',
        'ethnicity_asian',
        'ethnicity_hispanic',
        'ethnicity_latin',
        'gender_nonbinary',
        'occupation_healthcare',
        'occupation_influencer',
        'political_anticonservative',
        'political_antiliberal',
        'political_blm',
        'relationship_sibling',
        'religion_atheism',
        'religion_hinduism']:
        aux_load_dir=None
        
    uid2text = collect_training_tweets(tweet_type, uids, load_dir, aux_load_dir)
    
    # save to file
    print(f'Writing tsv file for {tweet_type} / {identity}')
    with gzip.open(join(save_dir,f'{tweet_type}.{identity}.tsv.gz'),'wt') as f:
        for uid,texts in tqdm(uid2text.items()):
            label = uid2label[uid]
            
            # write to file, a user can be represented as a max of "max_samples_per_user" times
            if len(texts)>=max_samples_per_user*n_tweets_per_sample:
                texts = sample(texts, max_samples_per_user*n_tweets_per_sample)
            while(len(texts)>=n_tweets_per_sample):
                line = ' '.join(texts[:n_tweets_per_sample])
                f.write(f'{uid}\t{label}\t{line}\n')
                texts = texts[n_tweets_per_sample:]    
    return

def merge_training_sets(user_id_file, load_dir, aux_load_dir, save_dir, n_tweets_per_sample=5, max_samples_per_user=5):
    """Loads all tweet files, and sorts them by each user, then by identity category

    Args:
        user_id_file (_type_): _description_
        load_dir (_type_): _description_
        save_dir (_type_): _description_
    """
    # get (sub)identity types and tweet types
    df = pd.read_csv(user_id_file,sep='\t',dtype={'user_id':str})
    identities = sorted(df.identity.unique())
    tweet_types = ['retweets_quotes','tweets_replies']
    
    all_inputs = []
    for tweet_type in tweet_types:
        for identity in identities:
            all_inputs.append((tweet_type,identity,user_id_file,load_dir,aux_load_dir,save_dir,n_tweets_per_sample,max_samples_per_user))
    
    pool = Pool(30)
    pool.starmap(merge_training_sets_worker, all_inputs)
    # merge_training_sets_worker(*all_inputs[0])
    
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
    aux_load_dir = '/shared/3/projects/bio-change/data/interim/identity_classifier-train_data/additional-tweets-from-api'
    save_dir = '/shared/3/projects/bio-change/data/interim/identity_classifier-train_data/train_data/5_tweets_per_sample-5_max_samples_per_user'
    merge_training_sets(user_id_file, load_dir, aux_load_dir, save_dir, n_tweets_per_sample=5, max_samples_per_user=5)