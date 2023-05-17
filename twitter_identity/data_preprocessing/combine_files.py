"""A file that contains the various scripts for merging different files
"""

import os
from os.path import join
import gzip
import re
from multiprocessing import Pool
from random import sample, shuffle
from time import time
import sys

from sklearn.utils import resample
import pandas as pd
import ujson as json
from tqdm import tqdm
# from ftlangdetect import detect

# from twitter_identity.utils.utils import write_data_file_info, strip_tweet

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
    if '/' in identity:
        identity=identity.replace('/','')
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

def split_data_train_test_val_worker(load_file, load_dir, save_dir, n_train_samples=50000):
    # get all tweets for each user so that the same user doesn't appear in multiple splits
    file_prefix = '.'.join(load_file.split('.')[:2]) # prefix for saving file
    
    # upsampled to 100K (80K/10K/10K) samples, balanced between two sets
    label2data = {'0':{}, '1':{}} # maps the tweets of all user by label
    # all_users = {} # users for train/test mapping
    
    # load all data
    with gzip.open(join(load_dir,load_file),'rt') as f:
        for line in f:
            uid,label,text=line.split('\t')
            if uid not in label2data[label]:
                label2data[label][uid]=[]
            label2data[label][uid].append(line)
            
    # split users into train/test/val
    user_assignment = {
        '1':{'train':[], 'val':[], 'test':[]},
        '0':{'train':[], 'val':[], 'test':[]}
        }
    for label,D in label2data.items():
        uids=list(D.keys())
        shuffle(uids)
        ln=int(0.1*len(uids))
        user_assignment[label]['test']=uids[:ln]
        user_assignment[label]['val']=uids[ln:ln*2]
        user_assignment[label]['train']=uids[ln*2:]
    
    split2count = {}
    
    # get tweet data for train/test/val
    for split in ['test','val']:
        all_samples = []
        for label in ['0','1']:
            split_samples = []
            uids = user_assignment[label][split]
            for uid in uids:
                split_samples.extend(label2data[label][uid]) # append all tweets by this user
            split2count[(split,label)]=len(split_samples)
            all_samples.extend(split_samples)

        # shuffle(all_samples)
        # print(split,label,len(all_samples))
        with gzip.open(join(save_dir,f'{file_prefix}.{split}.tsv.gz'),'wt') as f:
            for line in all_samples:
                f.write(line)
                
    # we have to match number of lines for train
    all_samples = []
    for label,D in user_assignment.items():
        split_samples = []
        uids = D['train']
        if len(uids)>=n_train_samples: # if uids>=train samples required, we just use one sample per uid
            uids2 = resample(uids,replace=False,n_samples=n_train_samples)
            for uid in uids2:
                split_samples.append(label2data[label][uid][0])
        else: # we augment one per user and check if we get over the limit
            S=set(uids)
            while(len(S)>0):
                S2 = set() # for next round
                uids=list(S)
                for uid in uids:
                    if len(split_samples)>=n_train_samples:
                        break
                    else:
                        split_samples.append(label2data[label][uid][0]) # add sample
                        label2data[label][uid].pop(0) # pop first item
                        if len(label2data[label][uid])>0:
                            S2.add(uid)
                if len(split_samples)>=n_train_samples:
                    break
                S = S2 # update valid users for next round
            if len(split_samples)>=n_train_samples:
                split_samples = split_samples[:n_train_samples]
            else:
                split_samples = resample(split_samples, replace=True, n_samples=n_train_samples)

        all_samples.extend(split_samples) # add to entire set
        split2count[('train',label)]=len(split_samples)

    print(file_prefix,split2count)
    with gzip.open(join(save_dir,f'{file_prefix}.train.tsv.gz'),'wt') as f:
        for line in all_samples:
            f.write(line)
    
    print(f"Wrote train set for {file_prefix}")
    return

def split_data_train_test_val(load_dir, save_dir, n_train_samples=50000):
    files = sorted(os.listdir(load_dir))
    inputs = [(file,load_dir,save_dir, n_train_samples) for file in files]
    pool = Pool(30)
    pool.starmap(split_data_train_test_val_worker, inputs)
    # split_data_train_test_val_worker(*inputs[0])
    write_data_file_info(__file__,split_data_train_test_val.__name__,save_dir,[load_dir])
    return

def merge_user_files(user_dir,save_dir):
    uid2info = {}
    files = sorted(os.listdir(user_dir))
    pbar = tqdm(files)
    for file in pbar:
        dt = file.split('.')[2]
        with gzip.open(join(user_dir,file),'rt') as f:
            for line in f:
                obj=json.loads(line)
                uid=obj['id_str']
                if uid not in uid2info:
                    obj['current_data']=dt
                    uid2info[uid] = obj
        pbar.set_description(f'{len(uid2info)} users')
    
    outf = gzip.open(join(save_dir,'all_user_profiles.json.gz'),'wt')
    for obj in tqdm(uid2info.values()):
        outf.write(json.dumps(obj)+'\n')
    outf.close()
    return

from datetime import datetime
from dateutil.parser import parse
import numpy as np
def get_weekly_bins(timestamp):
    """
    A function that returns the number of the week based on starting date (2020.04.01)
    :param timestamp: the current timestamp
    :return:
    """
    dt_base = datetime(2020, 4, 1)
    try:
        dt_current = datetime.fromtimestamp(float(timestamp))
    except:
        dt_current = parse(timestamp)
    dt_current = dt_current.replace(tzinfo=None)
    diff = dt_current - dt_base
    try:
        diff = dt_current - dt_base
    except:
        print('dt-current',dt_current)
        print('dt-base',dt_base)
        print(timestamp)
        import sys
        sys.exit(0)
    return int(np.floor(diff.days / 7))

def get_tweet_activity_worker(uid2week, tweet_file, save_file, weeks_prior=12, weeks_post=13):
    # ran on greatlakes
    # ran on taco
    
    start = time()

    uid2tweets = {uid:[] for uid in uid2week.keys()} # when uid posted something or responded to someone
    uid2origins = {uid:[] for uid in uid2week.keys()} # when uid was the origin of someone's response
    
    cnt=0
    all_tids = set()
    with gzip.open(tweet_file,'rt') as f:
        for line in f:
            obj=json.loads(line)
            tid = obj['id']
            if tid in all_tids:
                continue # one tweet is only considered once
            if 'user_id' in obj:
                uid=obj['user_id']
                if uid in uid2tweets:
                    week1 = uid2week[uid]
                    week2 = get_weekly_bins(obj['created_at'])
                    wd = week2-week1
                    if (wd>=-weeks_prior) and (wd<=weeks_post):
                        uid2tweets[uid].append(obj)
                        all_tids.add(tid)
                        cnt+=1
                        
            if 'user_id_origin' in obj:
                uid=obj['user_id_origin']
                if uid in uid2origins:
                    week1 = uid2week[uid]
                    week2 = get_weekly_bins(obj['created_at'])
                    wd = week2-week1
                    if (wd>=-weeks_prior) and (wd<=weeks_post):
                        uid2origins[uid].append(obj)
                        all_tids.add(tid)
                        cnt+=1
    
    # write tweets
    save_file = tweet_file.split('/')[-1].replace('tweets.','')
    print(f'Saving {save_file}!')
    with gzip.open(join(save_dir,'activities_made',save_file),'wt') as f:
        for uid,V in tqdm(uid2tweets.items()):
            for obj in V:
                f.write(json.dumps(obj)+'\n')
                
    with gzip.open(join(save_dir,'activities_origin',save_file),'wt') as f:
        for uid,V in tqdm(uid2origins.items()):
            for obj in V:
                f.write(json.dumps(obj)+'\n')
    print(f'{len(all_tids)} tweets for {save_file} {int(time()-start)}')
    # write_data_file_info(__file__, get_tweet_activity.__name__, save_dir, [user_file,tweet_dir])
    return
    
def get_tweet_activity(user_id_file, tweet_dir, save_dir, weeks_prior=12, weeks_post=12):
    # get users and their dates
    # df1=pd.DataFrame()
    # for typ in ['with_tweet_identity','without_tweet_identity']:
    #     # for cat in os.listdir(join(user_data_dir,typ)):
    #     files=[file for file in os.listdir(join(user_data_dir,typ)) if file.startswith('all_covariates')]
    #     for file in files:
    #         df2=pd.read_csv(join(user_data_dir,typ,file),sep='\t',dtype={'user_id':str})
    #         df1=pd.concat([df1,df2],axis=0)
    # df1=df1[['user_id','week_treated']].drop_duplicates()
    # uid2week={uid:wt for uid,wt in df1.values}
    # get all uids and the week of their profile change
    
    
    # uid2week = {}
    # df_uid=pd.read_csv(user_id_file,sep='\t',dtype={'user_id':str})
    # for uid,timestamp in tqdm(df_uid[['user_id','timestamp_treated']].values):
    #     uid2week[uid]=get_weekly_bins(timestamp)
    # print(len(uid2week),' users!')
    
    # get all uids and the week of their profile change
    uid2week = {}
    df_uid=pd.read_csv(user_id_file,sep='\t',dtype={'user_id':str})
    for uid,timestamp in tqdm(df_uid[['user_id','timestamp_treated']].values):
        uid2week[uid]=get_weekly_bins(timestamp)
    print(len(uid2week),' users!')

    
    # get files to load
    files = sorted([join(tweet_dir,file) for file in os.listdir(tweet_dir) if file.startswith('tweets.')])
    inputs = []
    for tweet_file in files:
        inputs.append((uid2week, tweet_file, save_dir, weeks_prior, weeks_post))
        
    pool = Pool(18)
    pool.starmap(get_tweet_activity_worker, inputs)
    write_data_file_info(__file__, get_tweet_activity.__name__, save_dir, [user_id_file,tweet_dir])
    # write_data_file_info(__file__, get_tweet_activity.__name__, save_dir, [user_data_dir,tweet_dir])
    # get_tweet_activity_worker(*inputs[100])
    return

def get_active_tweets_by_identity_worker(uid2week, activity_type, tweet_data_dir, save_file):
    # load valid users
    print(f'Starting {save_file}')
    
    # iterate through all files and record text as well as week differences
    uid2tweets = {uid:[] for uid in uid2week.keys()} 
    all_tids = set()
    files = [join(tweet_data_dir,activity_type,file) for file in sorted(os.listdir(join(tweet_data_dir,activity_type)))]
    for file in tqdm(files):
        with gzip.open(file,'rt') as f:
            for line in f:
                obj=json.loads(line)
                tid = obj['id']
                if tid in all_tids:
                    continue
                if activity_type=='activities_made':
                    uid=obj['user_id']
                elif activity_type=='activities_origin':
                    uid=obj['user_id_origin']
                if uid in uid2tweets:
                    w1 = uid2week[uid]
                    w2 = get_weekly_bins(obj['created_at'])
                    text = strip_tweet(obj['text'],url='replace')
                    wd=w2-w1
                    # if (wd<=0): # to only get pre-shock tweets
                    result = detect(text=text, low_memory=False)
                    if (result['score']>=0.5) and (result['lang']=='en'):
                        is_english=True
                    else:
                        is_english=False
                    if is_english:
                        uid2tweets[uid].append((wd,tid,obj['tweet_type'],is_english,text))
                        all_tids.add(tid)
    
    # save to output
    out=[]
    for uid,V in tqdm(uid2tweets.items()):
        for week_diff,tid,tweet_type,is_english,text in V:
            out.append((uid,tid,week_diff,tweet_type,is_english,text))
    df=pd.DataFrame(out,columns=['user_id','tweet_id','week_diff','tweet_type','is_english','text'])
    df[(df.tweet_type=='tweet')|(df.tweet_type=='mention')|(df.tweet_type=='reply')].to_csv(save_file.replace('.df.tsv.gz','.tweet.df.tsv.gz'),sep='\t',compression='gzip',index=False)    
    df[(df.tweet_type=='retweet')|(df.tweet_type=='quote')].to_csv(save_file.replace('.df.tsv.gz','.retweet.df.tsv.gz'),sep='\t',compression='gzip',index=False)    
    print(f'Finished {activity_type} {save_file}')
    return

def get_active_tweets_by_identity(user_data_dir, tweet_data_dir, save_dir):
    """For each identity, stores the tweets by treated and control users into a compressed dataframe

    Args:
        user_id_file (_type_): _description_
        load_dir (_type_): _description_
        save_dir (_type_): _description_
    """
    
    # get identities
    identities = sorted([file.split('.')[1] for file in os.listdir(join(user_data_dir,'without_tweet_identity')) if file.startswith('all_covariates')])

    inputs = []
    
    for identity in identities:
        
        df1=pd.DataFrame()
        for typ in ['with_tweet_identity','without_tweet_identity']:
            # for cat in os.listdir(join(user_data_dir,typ)):
            df2=pd.read_csv(join(user_data_dir,typ,f'all_covariates.{identity}.df.tsv'),sep='\t',dtype={'user_id':str})
            df1=pd.concat([df1,df2],axis=0)
        df1=df1[['user_id','week_treated']].drop_duplicates()
        uid2week={uid:wt for uid,wt in df1.values}
        
        # # get users and their dates
        # df1=pd.DataFrame()
        # for cat in os.listdir(user_data_dir):
        #     df2=pd.read_csv(join(user_data_dir,cat,f'all_covariates.{identity}.df.tsv'),sep='\t',dtype={'user_id':str})
        #     df1=pd.concat([df1,df2],axis=0)
        # df1=df1[['user_id','week_treated']].drop_duplicates()
        # uid2week={uid:wt for uid,wt in df1.values}
        
        for activity_type in ['activities_made','activities_origin']:
            save_file = join(save_dir,f'{activity_type}.{identity}.df.tsv.gz')
            
            inputs.append((uid2week, activity_type, tweet_data_dir, save_file))
    
            
    pool = Pool(18)
    pool.starmap(get_active_tweets_by_identity_worker, inputs)
    write_data_file_info(__file__, get_active_tweets_by_identity.__name__, save_dir, [user_data_dir, tweet_data_dir])
    # get_active_tweets_by_identity_worker(*inputs[0])
    return

def get_pre_change_tweets(user_id_file, load_dir, save_dir):
    """Gathers all the activities that happened before the profile change, to be used as covariates

    Args:
        user_id_file (_type_): _description_
        load_dir (_type_): _description_
        save_dir (_type_): _description_
    """
    # get all uids and the week of their profile change
    uid2week = {}
    df_uid=pd.read_csv(user_id_file,sep='\t',dtype={'user_id':str})
    for uid,timestamp in tqdm(df_uid[['user_id','timestamp_treated']].values):
        uid2week[uid]=get_weekly_bins(timestamp)
    print(len(uid2week),' users!')
    
    # load all tweets and find the ones where it was -12 ~ -1 weeks relative to the identity change
        # iterate through all files and record text as well as week differences
    uid2tweets = {uid:[] for uid in uid2week.keys()} 
    all_tids = set()
    files = [join(load_dir,file) for file in sorted(os.listdir(load_dir))]
    # files = files[200:250]
    for file in tqdm(files):
        with gzip.open(file,'rt') as f:
            for line in f:
                obj=json.loads(line)
                tid = obj['id']
                if tid in all_tids:
                    continue
                uid=obj['user_id']
                if uid in uid2tweets:
                    w1 = uid2week[uid]
                    w2 = get_weekly_bins(obj['created_at'])
                    week_diff = w2-w1
                    if (week_diff>=-12) and (week_diff<=-1):
                        text = strip_tweet(obj['text'])
                        uid2tweets[uid].append((week_diff,obj['tweet_type'],text))
                        all_tids.add(tid)
    
    # save
    out = []
    
    cnt=0
    for uid,V in tqdm(uid2tweets.items()):
        if len(V):
            cnt+=1
            for v in V:
                out.append((uid,v[0],v[1],v[2]))
    print(f'{cnt}/{len(uid2tweets)} nonempty users!')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    df_out=pd.DataFrame(out,columns=['user_id','week_diff','tweet_type','text'])
    df1=df_out[(df_out.tweet_type=='reply')|(df_out.tweet_type=='mention')|(df_out.tweet_type=='tweet')]
    df1.to_csv(join(save_dir,'all_past_tweets.df.tsv.gz'),sep='\t',index=False)
    df2=df_out[(df_out.tweet_type=='retweet')|(df_out.tweet_type=='quote')]
    df2.to_csv(join(save_dir,'all_past_retweets.df.tsv.gz'),sep='\t',index=False)
    
    write_data_file_info(__file__, get_pre_change_tweets.__name__,save_dir, [user_id_file,load_dir])      
    return

def combine_matched_users(load_dir, save_file):
    """
    Combines the CEM-matched treated and control pairs
    """
    df=pd.DataFrame()
    for file in sorted(os.listdir(load_dir)):
        if file.endswith('.df.tsv'):
            df2=pd.read_csv(join(load_dir,file),sep='\t',dtype={'user_id':str})
            df2['identity']=file.split('.')[0]
            df=pd.concat([df,df2],axis=0)
    
    df.to_csv(save_file, sep='\t', index=False)
    write_data_file_info(__file__, combine_matched_users.__name__, save_file, [load_dir])
    return

def get_identity_by_week(identity):
    start = time()
    
    file_dir = '/scratch/drom_root/drom0/minje/bio-change/04.extract-identities/raw-outputs'
    save_dir = '/scratch/drom_root/drom0/minje/bio-change/04.extract-identities/identity-by-week'

    uid2data = {}
    # aggregate users to file
    for file in sorted(os.listdir(file_dir)):
        wc = get_weekly_bins(file.split('.')[2])
        with gzip.open(join(file_dir,file),'rt') as f:
            for line in f:
                line = line.strip().split('\t')
                uid = line[0]
                if uid not in uid2data:
                    uid2data[uid]=[]
                if len(line)==3:
                    candidates = []
                    for c1 in line[2].split('|'):
                        candidates.append(c1.split(':')[0])
                    if identity in candidates:
                        uid2data[uid].append((wc,1))
                    else:
                        uid2data[uid].append((wc,0))
        s = int(time()-start)
        print(f'{file}! {s} seconds!')

    with gzip.open(join(save_dir,f'{identity}.tsv.gz'),'wt') as outf:
        for uid,V in uid2data.items():
            V = V + [(68,0)] # add week after last week to end list
            V=sorted(list(set(V)))
            line_out = []
            ln = len(V)
            for i in range(ln-1):
                w1,v1=V[i]
                w2,v2=V[i+1]
                if v1==1:
                    line_out.append(f'{w1}_{w2}')

            if len(line_out):
                outf.write(uid+'\t'+','.join(line_out)+'\n')

    return

if __name__=='__main__':
    # merge the identity files
    # load_dir = '/shared/3/projects/bio-change/data/interim/description_changes/extracted/splitted'
    # save_dir = '/shared/3/projects/bio-change/data/interim/description_changes/extracted/'
    # merge_splitted_extracted_identities(load_dir, save_dir)
    
    # create training data out of the collected tweets
    # user_id_file = '/shared/3/projects/bio-change/data/interim/identity_classifier-train_data/positive-negative-users/labels_200000.df.tsv'
    # load_dir = '/shared/3/projects/bio-change/data/raw/identity_classifier-train_data'
    # aux_load_dir = '/shared/3/projects/bio-change/data/interim/identity_classifier-train_data/additional-tweets-from-api'
    # save_dir = '/shared/3/projects/bio-change/data/interim/identity_classifier-train_data/train_data/5_tweets_per_sample-5_max_samples_per_user'
    # merge_training_sets(user_id_file, load_dir, aux_load_dir, save_dir, n_tweets_per_sample=5, max_samples_per_user=5)
    
    # split into train/test/val splits
    # load_dir = '/shared/3/projects/bio-change/data/interim/identity_classifier-train_data/train_data/5_tweets_per_sample-5_max_samples_per_user'
    # save_dir = '/shared/3/projects/bio-change/data/processed/identity_classifier-train_data'
    # split_data_train_test_val(load_dir, save_dir, 50000)
    
    # merge user info files
    # user_dir = '/shared/3/projects/bio-change/data/external/treated-control-tweets/users'
    # save_dir = '/shared/3/projects/bio-change/data/external/treated-control-tweets'
    # merge_user_files(user_dir, save_dir)
    
    # gets activities of valid users ranged by weeks since profile update
    # user_data_dir='/shared/3/projects/bio-change/data/processed/matching/matched-users'
    # user_data_dir='/shared/3/projects/bio-change/data/processed/matching/matched-users'
    # user_id_file= '/shared/3/projects/bio-change/data/interim/propensity-score-matching/description_change_features.df.tsv' # to obtain both treated & control (not matched) users
    # tweet_dir = '/shared/3/projects/bio-change/data/raw/treated-control-tweets/tweets'
    # save_dir= '/shared/3/projects/bio-change/data/raw/treated-control-tweets/activity_around_profile_update'
    # get_tweet_activity(user_id_file, tweet_dir, save_dir, 12, 12)
    
    # get tweets by identity
    # tweet_data_dir='/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/identity_added-with_tweet_identity/activity_around_profile_update'
    # save_dir='/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/identity_added-without_tweet_identity/tweets_by_identity'
    # user_data_dir='/shared/3/projects/bio-change/data/processed/matching/matched-users'
    # user_data_dir='/shared/3/projects/bio-change/data/interim/propensity-score-matching/all-matches/propensity/'
    # tweet_data_dir= '/shared/3/projects/bio-change/data/raw/treated-control-tweets/activity_around_profile_update'
    # save_dir='/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/tweets_by_identity'
    # get_active_tweets_by_identity(user_data_dir, tweet_data_dir, save_dir)

    # # user_id_file= '/scratch/drom_root/drom0/minje/bio-change/01.treated-control-users/description_features.df.tsv'
    # # load_dir='/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/activity_around_profile_update/activities_made'
    # user_id_file= '/shared/3/projects/bio-change/data/interim/propensity-score-matching/description_change_features.df.tsv'
    # load_dir='/shared/3/projects/bio-change/data/raw/treated-control-tweets/activity_around_profile_update/activities_made'
    # save_dir='/shared/3/projects/bio-change/data/interim/propensity-score-matching/past_tweets/'
    # get_pre_change_tweets(user_id_file, load_dir, save_dir)

    # load_dir='/shared/3/projects/bio-change/data/interim/propensity-score-matching/all-matches/cem'
    # save_file='/shared/3/projects/bio-change/data/interim/propensity-score-matching/all-matches/cem_matches.df.tsv'
    # combine_matched_users(load_dir, save_file)

    identity_dir = '/scratch/drom_root/drom0/minje/bio-change/07.matched-user-tweets/0.propensity-users/change_added-with_text'
    identities = sorted([file.split('.')[1] for file in os.listdir(identity_dir) if file.startswith('all_covariates')])
    print('before',identities)
    if len(sys.argv)>=2:
        identities = [identity for idx,identity in enumerate(identities) if idx%10==int(sys.argv[1])]
    print('after',identities)
    for identity in identities:
        get_identity_by_week(identity)