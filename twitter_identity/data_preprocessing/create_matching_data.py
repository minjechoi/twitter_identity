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
    df3['week_diff']='week_diff_'+df3['week_diff']
    df4=df3.pivot(index='user_id',columns='week_diff',values='text')
    df5=df4.reset_index()
    save_file = join(save_dir,'past_activities_per_week_features.df.tsv')
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
            # fix for cath/christ
            id_=id_.replace('/','')
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
    df2 = df2[['user_id']+[f'week_diff_{i}' for i in range(-4,0)]]
    
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
    identities = [file.split('.')[1] for file in sorted(os.listdir(identity_dir)) if file.startswith('retweet-classifier')]
    
    identities = ['religion_cathchrist']
    pool = Pool(12)
    inputs = []
    for identity in identities:
        inputs.append((user_dir,base_dir,save_dir,identity))
    for X in inputs:
        combine_covariates_worker(*X)
    # pool.starmap(combine_covariates_worker,inputs)
    
    return

def propensity_matching_worker(cov_dir, save_dir, identity, include_text_score=False):
    """Manually runs propensity score matching

    Args:
        load_dir (_type_): _description_
        save_dir (_type_): _description_
        identity (_type_): _description_
    """
    from random import sample
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from lightgbm import LGBMClassifier
    from jenkspy import JenksNaturalBreaks
    import numpy as np
    from sklearn.metrics import f1_score,roc_auc_score,accuracy_score
    from sklearn.preprocessing import StandardScaler
    from statsmodels.stats.meta_analysis import effectsize_smd

    # load covariates
    df_cov=pd.read_csv(join(cov_dir,f'{identity}.df.tsv'),sep='\t',dtype={'user_id':str})
    
    # get propensity score on original data
    X = df_cov.drop(['is_identity'],axis=1)
    y = df_cov.is_identity
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    valid_covariates=X.columns.tolist()
    valid_covariates = [cov for cov in valid_covariates if cov not in ['user_id','week_treated']]
    if include_text_score==False:
        valid_covariates = [cov for cov in valid_covariates if not (cov.startswith('prev_tweet') or cov.startswith('prev_retweet'))]
    
    # cls = LGBMClassifier()
    cls = LogisticRegression(max_iter=1000, class_weight='balanced')
    cls.fit(X_train[valid_covariates], y_train)
    y_score=cls.predict_proba(X_test[valid_covariates])[:,1]
    y_pred=cls.predict(X_test[valid_covariates])
    auc = roc_auc_score(y_true=y_test, y_score=y_score)
    print('auc:%.3f'%auc)
    f1 = f1_score(y_true=y_test, y_pred=y_pred)
    print('f1:%.3f'%f1)
    with open(join(save_dir,f'performance.{identity}.txt'),'w') as f:
        f.write('auc:%.3f\n'%auc)
        f.write('f1:%.3f\n'%f1)
    
    # assign propensity scores and get natural breaks
    propensity_scores=cls.predict_proba(df_cov[valid_covariates])[:,1]
    n_classes=int(np.sqrt(df_cov.is_identity.sum()))
    data=sample(propensity_scores.tolist(),10000)
    jnb=JenksNaturalBreaks(n_classes)
    jnb.fit(data)
    strata=jnb.predict(propensity_scores)
    X['strata']=strata
    
    # run frobenius norm on matched data
    scaler=StandardScaler()
    X[valid_covariates]=scaler.fit_transform(X[valid_covariates])
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
    df2_cov=df.merge(df_cov,how='left')
    df2=pd.DataFrame(out2,columns=['user_treated','user_matched'])
    
    # save the files
    df2_cov.to_csv(join(save_dir,f'all_covariates.{identity}.df.tsv'),sep='\t',index=False)
    df2.to_csv(join(save_dir,f'matched_pairs.{identity}.df.tsv'),sep='\t',index=False)
    
    # see whether SMD decreased after matching
    out=[]
    for cov in valid_covariates:
        # original
        arr1,arr2=df_cov[df_cov.is_identity==1][cov],df_cov[df_cov.is_identity==0][cov]
        mean1,sd1,nobs1=np.mean(arr1),np.std(arr1),len(arr1)
        mean2,sd2,nobs2=np.mean(arr2),np.std(arr2),len(arr2)
        smd1,_=effectsize_smd(mean1, sd1, nobs1, mean2, sd2, nobs2)
        out.append(('original',cov,np.abs(smd1)))
        
        # matched
        arr1,arr2=df2_cov[df2_cov.is_identity==1][cov],df2_cov[df2_cov.is_identity==0][cov]
        mean1,sd1,nobs1=np.mean(arr1),np.std(arr1),len(arr1)
        mean2,sd2,nobs2=np.mean(arr2),np.std(arr2),len(arr2)
        smd2,_=effectsize_smd(mean1, sd1, nobs1, mean2, sd2, nobs2)
        out.append(('matched',cov,np.abs(smd2)))
    df3=pd.DataFrame(out,columns=['setting','covariate','smd'])
    
    df3.to_csv(join(save_dir,f'smd.{identity}.df.tsv'),sep='\t',index=False)
    print("Finished ",identity)
    return

def propensity_matching(cov_dir, save_dir, include_text_score=False):
    identity_dir = '/shared/3/projects/bio-change/data/interim/propensity-score-matching/past_tweets/identity-scores'
    identities = [file.split('.')[1] for file in sorted(os.listdir(identity_dir)) if file.startswith('retweet-classifier')]
    pool = Pool(12)
    inputs = []
    for identity in sorted(identities):
        inputs.append((cov_dir,save_dir,identity,include_text_score))    
    pool.starmap(propensity_matching_worker, inputs)
    # propensity_matching_worker(*inputs[0])
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

    # past_data_dir='/shared/3/projects/bio-change/data/interim/propensity-score-matching/past_tweets/'
    # save_dir = '/shared/3/projects/bio-change/data/interim/propensity-score-matching'
    # get_weekly_counts(valid_users, past_data_dir, save_dir)
    
    # user_dir = '/shared/3/projects/bio-change/data/interim/treated-control-users'
    # base_dir = '/shared/3/projects/bio-change/data/interim/propensity-score-matching'
    # save_dir = '/shared/3/projects/bio-change/data/interim/propensity-score-matching/all-covariates'
    # # identity = 'gender_nonbinary'
    # combine_covariates(user_dir,base_dir,save_dir)
    
    # cov_dir = '/shared/3/projects/bio-change/data/interim/propensity-score-matching/all-covariates'
    # save_dir = '/shared/3/projects/bio-change/data/interim/propensity-score-matching/all-matches/propensity/without_tweet_identity'
    # propensity_matching(cov_dir,save_dir,False)
    
    # cov_dir = '/shared/3/projects/bio-change/data/interim/propensity-score-matching/all-covariates'
    # save_dir = '/shared/3/projects/bio-change/data/interim/propensity-score-matching/all-matches/propensity/with_tweet_identity'
    # propensity_matching(cov_dir,save_dir,True)