import os
from os.path import join
import gzip
from datetime import datetime
from time import time
from multiprocessing import Pool

import pandas as pd
import ujson as json
from tqdm import tqdm
from dateutil.parser import parse

from twitter_identity.utils.utils import get_weekly_bins,write_data_file_info, week_diff_to_month_diff

def load_uids(treat_user_file, control_user_file):
    """_summary_

    Args:
        treat_user_file (_type_): _description_
        control_user_file (_type_): _description_
    """
    valid_users=set()
    with open(treat_user_file) as f:
        for line in f:
            line=line.split('\t')
            uid=line[1]
            valid_users.add(uid)
    with open(control_user_file) as f:
        for line in f:
            line=line.split('\t')
            uid=line[0]
            valid_users.add(uid)    
    return valid_users
    

def obtain_profile_features(uids, user_info_file, save_dir):
    """Gets profile features (num of friends/followers/statuses and creation date)

    Args:
        uids: a list or set containing treated/control users
        user_info_file (_type_): _description_
        save_dir (_type_): _description_
    """
    uid2info = {uid:None for uid in uids}
    print(f'{len(uids)} candidate users!')
    with gzip.open(user_info_file,'rt') as f:
        for line in f:
            obj = json.loads(line)
            uid=obj['id_str']
            if uid in uid2info:
                uid2info[uid] = {
                    'fri':obj['friends_count'],
                    'fol':obj['followers_count'],
                    'sta':obj['statuses_count'],
                    'created_at':obj['created_at'],
                    }
    out = []
    for uid,V in tqdm(uid2info.items()):
        if V is not None:
            out.append((uid, V['fri'], V['fol'], V['sta'], V['created_at']))
    df=pd.DataFrame(out,columns=['user_id','fri','fol','sta','created_at'])
    save_file = join(save_dir,'user_activity_features.df.tsv')
    df.to_csv(save_file,sep='\t',index=False)
    write_data_file_info(__file__, obtain_profile_features.__name__, save_file, [user_info_file])
    print(df.shape)
    print(f'Saved to {save_file}')
    return

def obtain_description_features(uids, desc_file, save_dir):
    """Gets (1) the previous description, (2) the time when the profile was changed, and (3) the week of (2)

    Args:
        uids (_type_): _description_
        desc_file (_type_): _description_
        save_dir (_type_): _description_
    """
    uid2info = {uid:[] for uid in uids}
    with gzip.open(desc_file,'rt') as f:
        for line in f:
            uid,dt,desc=line.split('\t')
            desc=desc.strip()
            if uid in uid2info:
                uid2info[uid].append((float(dt),desc))
    
    out = []
    for uid,V in tqdm(uid2info.items()):
        assert len(V)==2
        dt,desc = V[1][0], V[0][1]
        week = get_weekly_bins(dt)
        out.append((uid,dt,week,desc))
    df=pd.DataFrame(out,columns=['user_id','timestamp_treated','week_treated','profile_before_update'])
    save_file = join(save_dir,'description_features.df.tsv')
    write_data_file_info(__file__, obtain_description_features.__name__, save_file, [desc_file])
    df.to_csv(save_file,sep='\t',index=False)
    print(df.shape)
    print(f'Saved to {save_file}')
    return

def get_weekly_counts(uids, data_dir, save_dir):
    """Get the weekly counts for activity levels per user

    Args:
        user_info_file (_type_): _description_
        data_file (_type_): _description_
        save_file (_type_): _description_
    """
    # get default dataframe to append other data on
    uid_weeks = []
    for week in range(-4,0):
        for uid in uids:
            uid_weeks.append((uid,week))
    df_uids = pd.DataFrame(uid_weeks,columns=['user_id','week_diff'])    
    
    # load all interactions
    df2=pd.DataFrame()
    for typ in ['tweet','retweet']:
        df=pd.read_csv(join(data_dir,f'all_past_{typ}s.df.tsv.gz'),sep='\t',dtype={'user_id':str})
        df2=pd.concat([df2,df],axis=0)
    
    df3 = df2.groupby(['user_id','week_diff']).count().reset_index()
    df3 = df_uids.merge(df3,on=['user_id','week_diff'],how='left').fillna(0)

    # combine week & tweet type into a new column
    df3['week_diff']=df3['week_diff'].astype(str)
    df3['week_diff']=+'week_diff_'+df3['week_diff']
    df4=df3.pivot(index='user_id',columns='week_diff',values='text')
    df5=df4.reset_index()
    save_file = join(save_dir,'past_activities_per_week.df.tsv')
    df5.to_csv(save_file,sep='\t',index=False)
    write_data_file_info(__file__, get_weekly_counts.__name__, save_file, [data_dir])
    return


def combine_covariates_worker(user_dir, base_dir, save_dir, identity):
    """Merges all covariates into a dataframe -> we will use R-matchit to compute the propensity scores

    Args:
        user_dir (_type_): _description_
        base_dir (_type_): _description_
        save_dir (_type_): _description_
        identity (_type_): _description_
    """
    start = time()

    # get a list of treated and potential control users
    uid_pos = []
    with open(join(user_dir,'all_treated_users.tsv')) as f:
        for line in f:
            id_,uid,dt,phrase=line.split('\t')
            if id_==identity:
                uid_pos.append((uid,id_))
    df_pos = pd.DataFrame(uid_pos,columns=['user_id','identity'])
    uid_neg=[]
    with open(join(user_dir,'all_potential_control_users.tsv')) as f:
        for line in f:
            uid,dt=line.split('\t')
            uid_neg.append((uid,'None'))
    df_neg = pd.DataFrame(uid_neg,columns=['user_id','identity'])
    df_uid=pd.concat([df_pos,df_neg],axis=0)
    print(f'Got positive/negative users for {identity}! {int(start-time())}')
    
    # load relevant covariates
    df1=pd.read_csv(join(base_dir,'description_change_features.df.tsv'),sep='\t',dtype={'user_id':str})
    with open(join(base_dir,'profile-identity-scores',f'tweet-classifier.{identity}.txt')) as f:
        scores=[float(x) for x in f.readlines()]
    df1['profile_score']=scores
    df_uid=df_uid.merge(df1.drop(columns=['profile_before_update'],axis=1), on=['user_id'],how='inner')
    
    # remove both treated & control users who have high profile scores even before the change
    df_uid = df_uid[df_uid.profile_score<0.5]
    df_uid.groupby(['identity']).count()
    print(f'Got description change scores for {identity}! {int(start-time())}')
    
    df2=pd.read_csv(join(base_dir,'past_activities_per_week_features.df.tsv'),sep='\t',dtype={'user_id':str})
    # optional - merge the tweet type features by week
    for i in range(-4,0):
        df2[f'activities_{i}'] = df2[f'tweet_{i}'] + df2[f'retweet_{i}'] + df2[f'quote_{i}'] + df2[f'reply_{i}'] + df2[f'mention_{i}']
    df2 = df2[['user_id']+[f'activities_{i}' for i in range(-4,0)]]
    
    df_uid=df_uid.merge(df2,on=['user_id'],how='inner')
    print(f'Got past activity features for {identity}! {int(start-time())}')
    
    df3=pd.read_csv(join(base_dir,'user_profile_features.df.tsv'),sep='\t',dtype={'user_id':str})
    df_uid=df_uid.merge(df3,on=['user_id'],how='inner')
    n_days_since_profile = []
    for v1,v2 in tqdm(df_uid[['timestamp_treated','created_at']].values):
        dt1=datetime.fromtimestamp(v1)
        dt2=parse(v2,ignoretz=True)
        diff=(dt1-dt2).days
        n_days_since_profile.append(diff)
    df_uid['n_days_since_profile']=n_days_since_profile
    df_uid = df_uid.drop(columns=['timestamp_treated','created_at'],axis=1)
    print(f'Add user profile features for {identity}! {int(start-time())}')
    
    # add scores of past tweet and retweets with identity scores
    for tweet_type in ['tweet','retweet']:
        with open(join(base_dir,'past_tweets','identity-scores',f'{tweet_type}-classifier.{identity}.txt')) as f:
            scores=[float(x) for x in f.readlines()]

        df4=pd.read_csv(join(base_dir,'past_tweets',f'all_past_{tweet_type}s.df.tsv.gz'),sep='\t',dtype={'user_id':str})
        df4[f'{tweet_type}_score']=scores
        
        df5=df4[['user_id','week_diff',f'{tweet_type}_score']].groupby(['user_id','week_diff']).mean().reset_index()
        df5=df5.pivot(index='user_id',columns=['week_diff'],values=f'{tweet_type}_score').fillna(0).reset_index()
        df5.columns=['user_id']+[f'prev_{tweet_type}-mean_{i}' for i in range(-4,0)]

        df6=df4[['user_id','week_diff',f'{tweet_type}_score']].groupby(['user_id','week_diff']).max().reset_index()
        df6=df6.pivot(index='user_id',columns=['week_diff'],values=f'{tweet_type}_score').fillna(0).reset_index()
        df6.columns=['user_id']+[f'prev_{tweet_type}-max_{i}' for i in range(-4,0)]        

        df_uid=df_uid.merge(df5,on=['user_id'], how='left').fillna(0)
        df_uid=df_uid.merge(df6,on=['user_id'], how='left').fillna(0)

    df_uid['is_identity']=(df_uid.identity==identity).astype(int)
    df_uid = df_uid.drop(columns=['identity'],axis=1)
    df_uid.to_csv(join(save_dir,f'{identity}.df.tsv'),sep='\t',index=False)
    print(f'Completed saving covariates for {identity}! {int(start-time())}')
    return

def combine_covariates(user_dir, base_dir, save_dir):
    identity_dir = '/shared/3/projects/bio-change/data/interim/propensity-score-matching/past_tweets/identity-scores'
    # identities = [file.split('.')[1] for file in sorted(os.listdir(identity_dir)) if file.startswith('retweet-classifier')]
    identities = ['age_13-17']
    pool = Pool(12)
    inputs = []
    for identity in identities:
        inputs.append((user_dir,base_dir,save_dir,identity))
    for X in inputs:
        combine_covariates_worker(*X)
    # pool.starmap(combine_covariates_worker,inputs)
    
    return

    

if __name__=='__main__':
    treat_user_file = '/shared/3/projects/bio-change/data/interim/treated-control-users/all_treated_users.tsv'
    control_user_file = '/shared/3/projects/bio-change/data/interim/treated-control-users/all_potential_control_users.tsv'
    save_dir = '/shared/3/projects/bio-change/data/interim/propensity-score-matching'
    user_info_file = '/shared/3/projects/bio-change/data/external/treated-control-tweets/all_user_profiles.json.gz'

    valid_users = load_uids(treat_user_file, control_user_file)
    # obtain_profile_features(valid_users, user_info_file, save_dir)
    # desc_info_file = '/shared/3/projects/bio-change/data/interim/description_changes/filtered/description_changes_1plus_changes.tsv.gz'
    # obtain_description_features(valid_users, desc_info_file, save_dir)

    past_data_dir='/shared/3/projects/bio-change/data/interim/propensity-score-matching/past_tweets/'
    save_dir = '/shared/3/projects/bio-change/data/interim/propensity-score-matching'
    get_weekly_counts(valid_users, past_data_dir, save_dir)
    
    # user_dir = '/shared/3/projects/bio-change/data/interim/treated-control-users'
    # base_dir = '/shared/3/projects/bio-change/data/interim/propensity-score-matching'
    # save_dir = '/shared/3/projects/bio-change/data/interim/propensity-score-matching/all-covariates'
    # identity = 'gender_nonbinary'
    # combine_covariates(user_dir,base_dir,save_dir)