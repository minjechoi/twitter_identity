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
import numpy as np
from statsmodels.stats.meta_analysis import effectsize_smd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from jenkspy import JenksNaturalBreaks
from sklearn.metrics import f1_score,roc_auc_score
from random import sample

from twitter_identity.utils.utils import get_weekly_bins,write_data_file_info, week_diff_to_month_diff, get_identities

# Returns the user ids of treated and potential control users
def load_uids():
    data_dir='/shared/3/projects/bio-change/data/interim/description_changes/06.by-change-type'
    # def load_uids(data_dir):
    uids=set()
    for file in os.listdir(data_dir):
        if file.startswith('1_change'):
            df=pd.read_csv(join(data_dir,file),sep='\t',dtype={'user_id':str})
            uids.update(df['user_id'])
    return uids

# Gets profile features (num of friends/followers/statuses and creation date)
def obtain_profile_features(uids, user_info_file, save_dir):
    """
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

# Gets previous description
def obtain_description_features(user_dir, save_file):
    df=pd.DataFrame()
    for file in os.listdir(user_dir):
        if file.startswith('1_change'):
            df2=pd.read_csv(join(user_dir,file),sep='\t',dtype={'user_id':str})
            df=pd.concat([df,df2],axis=0)
    df['week_treated']=df['dt_after'].apply(get_weekly_bins)
    df=df[['user_id','week_treated','desc_before']]
    df.columns=['user_id','week_treated','text']
    df=df.drop_duplicates()
    df.to_csv(save_file,sep='\t',index=False)
    write_data_file_info(__file__, obtain_description_features.__name__, save_file, [user_dir])
    return

def get_weekly_counts(data_dir, save_file):
    """Get the weekly counts for activity levels per user

    Args:
        data_file (_type_): _description_
        save_file (_type_): _description_
    """
    # load all interactions
    df1=pd.read_csv(join(data_dir,'tweets_replies.4_weeks.df.tsv'),sep='\t',dtype={'user_id':str})
    df1 = df1.groupby('user_id').count().reset_index()[['user_id','text']]
    df1.columns = ['user_id','n_tweets']
    df2=pd.read_csv(join(data_dir,'retweets_quotes.4_weeks.df.tsv'),sep='\t',dtype={'user_id':str})
    df2 = df2.groupby('user_id').count().reset_index()[['user_id','text']]
    df2.columns = ['user_id','n_retweets']
    
    df3=df1.merge(df2,on=['user_id'],how='outer').fillna(0)
    df3.to_csv(save_file,sep='\t',index=False)
    write_data_file_info(__file__, get_weekly_counts.__name__, save_file, [data_dir])
    return

def propensity_matching_worker(save_dir, identity, change='added', include_text_score=False):
    """
    change: either ['added','removed'], to indicate whether a user added or removed an identity phrase
    """
    # update save directory
    save_dir = join(save_dir,f'change_{change}-'+['without_text','with_text'][int(include_text_score)])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    if os.path.exists(join(save_dir,f'smd.{identity}.df.tsv')):
        print("Skipping ",identity)
        return
        
    # get list of treated and control users
    user_dir='/shared/3/projects/bio-change/data/interim/propensity-score-matching/users/before-matching'
    if change=='added':
        df_pos=pd.read_csv(join(user_dir,'users_01.df.tsv'),sep='\t',dtype={'user_id':str})
        df_neg=pd.read_csv(join(user_dir,'users_00.df.tsv'),sep='\t',dtype={'user_id':str})
    elif change=='removed':
        df_pos=pd.read_csv(join(user_dir,'users_10.df.tsv'),sep='\t',dtype={'user_id':str})
        df_neg=pd.read_csv(join(user_dir,'users_11.df.tsv'),sep='\t',dtype={'user_id':str})

    df_pos['week_treated']=df_pos['dt_after'].apply(get_weekly_bins)
    df_neg['week_treated']=df_neg['dt_after'].apply(get_weekly_bins)
    df_pos=df_pos[['user_id','identity','week_treated']]
    df_neg=df_neg[['user_id','week_treated']]
    df_pos2=df_pos[df_pos.identity==identity][['user_id','week_treated']]
    df_pos2['label']=1
    df_neg['label']=0
    df_cov=pd.concat([df_pos2,df_neg],axis=0)
    
    # merge with covariates
    df=pd.read_csv('/shared/3/projects/bio-change/data/interim/propensity-score-matching/covariates/past-tweet-count/past_1_month.tweets_and_retweets.df.tsv',sep='\t',dtype={'user_id':str})
    df_cov=df_cov.merge(df,on=['user_id'],how='inner')

    df=pd.read_csv('/shared/3/projects/bio-change/data/interim/propensity-score-matching/covariates/profile-stats/stats.df.tsv',sep='\t',dtype={'user_id':str})
    df_cov=df_cov.merge(df.drop(columns=['created_at'],axis=1),on=['user_id'],how='inner')    
    
    df=pd.read_csv('/shared/3/projects/bio-change/data/interim/propensity-score-matching/covariates/past-tweet-scores/english_tweets-4_weeks.df.tsv.gz',
                    sep='\t',dtype={'user_id':str},usecols=['user_id'])
    with open(f'/shared/3/projects/bio-change/data/interim/propensity-score-matching/covariates/past-tweet-scores/identity-scores/tweet-classifier.{identity}.txt') as f:
        scores=[float(x) for x in f.readlines()]
    df['tweet_identity_score']=scores
    df=df[df.tweet_identity_score>=0.5].groupby('user_id').count().reset_index()
    df_cov=df_cov.merge(df,on=['user_id'],how='left').fillna(0)

    df=pd.read_csv('/shared/3/projects/bio-change/data/interim/propensity-score-matching/covariates/profile-scores/desc_before.df.tsv',
                    sep='\t',dtype={'user_id':str},usecols=['user_id'])
    with open(f'/shared/3/projects/bio-change/data/interim/propensity-score-matching/covariates/profile-scores/identity-scores/tweet-classifier.{identity}.txt') as f:
        scores=[float(x) for x in f.readlines()]
    df['profile_identity_score']=scores
    df_cov=df_cov.merge(df,on=['user_id'],how='left').fillna(0) # all covariates
    
    X = df_cov.drop(['label'],axis=1)
    y = df_cov.label
    valid_covariates=X.columns.tolist()
    valid_covariates = [cov for cov in valid_covariates if cov not in ['user_id','week_treated']]
    if include_text_score==False:
        valid_covariates = [cov for cov in valid_covariates if cov!='tweet_identity_score']
    scaler=StandardScaler()
    X[valid_covariates]=scaler.fit_transform(X[valid_covariates])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cls = LogisticRegression(max_iter=1000, class_weight='balanced')
    cls.fit(X_train[valid_covariates], y_train)
    y_score=cls.predict_proba(X_test[valid_covariates])[:,1]
    y_pred=cls.predict(X_test[valid_covariates])
    auc = roc_auc_score(y_true=y_test, y_score=y_score)
    print('auc:%.3f'%auc)
    f1 = f1_score(y_true=y_test, y_pred=y_pred)
    print('f1:%.3f'%f1)

    propensity_scores=cls.predict_proba(df_cov[valid_covariates])[:,1]
    n_classes=int(np.sqrt(df_cov.label.sum()))
    # n_classes=10
    data=sample(propensity_scores.tolist(),10000)
    jnb=JenksNaturalBreaks(n_classes)
    jnb.fit(data)
    strata=jnb.predict(propensity_scores)
    X['strata']=strata
    X['label']=y

    pos_strata=sorted(X[X.label==1].strata.unique())
    pos2matches={}
    k=5
    for s in tqdm(pos_strata):
        X2=X[X.strata==s]
        weeks=sorted(X2[X2.label==1].week_treated.unique())
        for w in weeks:
            X3=X2[X2.week_treated==w]
            X3_pos=X3[X3.label==1]
            X3_neg=X3[X3.label==0]
            if len(X3_neg)>0:
                # get arrays
                arr_pos=X3_pos.drop(['user_id','week_treated','label','strata'],axis=1).to_numpy()
                uid_pos=X3_pos.user_id.tolist()
                arr_neg=X3_neg.drop(['user_id','week_treated','label','strata'],axis=1).to_numpy()
                uid_neg=X3_neg.user_id.tolist()
                # get up to 5 matches for each user
                for i,uid in enumerate(uid_pos):
                    if len(uid_neg)<=5:
                        # if running short, just use all available negative samples
                        pos2matches[uid]=uid_neg
                    else:
                        # if not, use 
                        vec_pos=arr_pos[i]
                        # compute the distance between a positive vector and all negative arrays
                        distances=np.linalg.norm(arr_neg - vec_pos, axis=1)
                        topk = sorted([(x,i) for i,x in enumerate(distances)])[:k]
                        pos2matches[uid]=[uid_neg[i] for x,i in topk]

    # get all covariates of the users (note: control users can appear multiple times as they can be matched multiple times)
    out=[]
    out2=[]
    for uid_pos,V in pos2matches.items():
        out.append(uid_pos)
        for uid_neg in V:
            out.append(uid_neg)
            out2.append((uid_pos,uid_neg))
    df=pd.DataFrame(out,columns=['user_id'])

    # get the covariates of the matched pairs, to compare with the original covariates
    df2_cov=df.merge(df_cov,how='inner')
    df_pairs=pd.DataFrame(out2,columns=['user_treated','user_matched'])
    
    df2_cov.to_csv(join(save_dir,f'all_covariates.{identity}.df.tsv'),sep='\t',index=False)
    df_pairs.to_csv(join(save_dir,f'pairs.{identity}.df.tsv'),sep='\t',index=False)

    # see whether SMD decreased after matching
    out=[]
    for cov in valid_covariates:
        # original
        arr1,arr2=df_cov[df_cov.label==1][cov],df_cov[df_cov.label==0][cov]
        mean1,sd1,nobs1=np.mean(arr1),np.std(arr1),len(arr1)
        mean2,sd2,nobs2=np.mean(arr2),np.std(arr2),len(arr2)
        smd1,_=effectsize_smd(mean1, sd1, nobs1, mean2, sd2, nobs2)
        out.append(('original',cov,np.abs(smd1)))

        # matched
        arr1,arr2=df2_cov[df2_cov.label==1][cov],df2_cov[df2_cov.label==0][cov]
        mean1,sd1,nobs1=np.mean(arr1),np.std(arr1),len(arr1)
        mean2,sd2,nobs2=np.mean(arr2),np.std(arr2),len(arr2)
        smd2,_=effectsize_smd(mean1, sd1, nobs1, mean2, sd2, nobs2)
        out.append(('matched',cov,np.abs(smd2)))
    df3=pd.DataFrame(out,columns=['setting','covariate','smd'])
    df3.to_csv(join(save_dir,f'smd.{identity}.df.tsv'),sep='\t',index=False)
    return


def combine_covariates(user_dir, base_dir, save_dir):
    identities = get_identities()

    pool = Pool(12)
    inputs = []
    for identity in identities:
        inputs.append((user_dir,base_dir,save_dir,identity))
    for X in inputs:
        combine_covariates_worker(*X)
    # pool.starmap(combine_covariates_worker,inputs)
    
    return

def propensity_matching(save_dir, include_text_score=False):
    # identity_dir = '/shared/3/projects/bio-change/data/interim/propensity-score-matching/past_tweets/identity-scores'
    # identities = [file.split('.')[1] for file in sorted(os.listdir(identity_dir)) if file.startswith('retweet-classifier')]
    identities = get_identities()
    pool = Pool(12)
    inputs = []
    for identity in sorted(identities):
        # inputs.append((save_dir,identity,include_text_score))    
        propensity_matching_worker(save_dir,identity,include_text_score)

    # pool.starmap(propensity_matching_worker, inputs)
    # propensity_matching_worker(*inputs[0])
    return



if __name__=='__main__':
    # treat_user_file = '/shared/3/projects/bio-change/data/interim/treated-control-users/all_treated_users.tsv'
    # control_user_file = '/shared/3/projects/bio-change/data/interim/treated-control-users/all_potential_control_users.tsv'

    # 1) get user stats
    # valid_users = load_uids()
    # user_info_file='/shared/3/projects/bio-change/data/raw/user_info/user_profile-2020.04.json.gz'
    # save_dir = '/shared/3/projects/bio-change/data/interim/propensity-score-matching/covariates/profile-stats'
    # obtain_profile_features(valid_users, user_info_file, save_dir)
    
    # 2)  get descriptions of users
    # user_dir='/shared/3/projects/bio-change/data/interim/description_changes/06.by-change-type'
    # save_file = '/shared/3/projects/bio-change/data/interim/propensity-score-matching/covariates/profile-scores/desc_before.df.tsv'
    # obtain_description_features(user_dir,save_file)
    
    # 3) get -4-1 week tweets of the users
    
    
    # desc_info_file = '/shared/3/projects/bio-change/data/interim/description_changes/filtered/description_changes_1plus_changes.tsv.gz'
    # obtain_description_features(valid_users, desc_info_file, save_dir)

    past_data_dir='/shared/3/projects/bio-change/data/interim/propensity-score-matching/covariates/past-tweets/raw-tweets'
    save_file = '/shared/3/projects/bio-change/data/interim/propensity-score-matching/covariates/past-tweets/past-tweet-count/past_1_month.tweets_and_retweets.df.tsv'
    get_weekly_counts(past_data_dir, save_file)
    
    # user_dir = '/shared/3/projects/bio-change/data/interim/treated-control-users'
    # base_dir = '/shared/3/projects/bio-change/data/interim/propensity-score-matching'
    # save_dir = '/shared/3/projects/bio-change/data/interim/propensity-score-matching/all-covariates'
    # # identity = 'gender_nonbinary'
    # combine_covariates(user_dir,base_dir,save_dir)
    
    # save_dir = '/shared/3/projects/bio-change/data/interim/propensity-score-matching/users/after-matching/without_tweet_identity'
    # propensity_matching(save_dir,False)
    # save_dir = '/shared/3/projects/bio-change/data/interim/propensity-score-matching/users/after-matching/with_tweet_identity'
    # propensity_matching(save_dir,True)
    
    # cov_dir = '/shared/3/projects/bio-change/data/interim/propensity-score-matching/all-covariates'
    # save_dir = '/shared/3/projects/bio-change/data/interim/propensity-score-matching/all-matches/propensity/with_tweet_identity'
    # propensity_matching(cov_dir,save_dir,True)