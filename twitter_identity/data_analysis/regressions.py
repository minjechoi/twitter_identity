import os
from os.path import join
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2


# added due to greatlakes
def get_identities():
    """Simply returns the list of identities
    """
    user_data_dir='/scratch/drom_root/drom0/minje/bio-change/06.regression/cov_dir'
    identities = sorted([file.split('.')[1] for file in os.listdir(user_data_dir) if file.startswith('all_covariates')])
    return identities

def week_diff_to_month_diff(week):
    """Switches week difference to month difference. Month difference here is actually 4-weeks

    Args:
        week (_type_): _description_

    Returns:
        _type_: _description_
    """
    if week==0:
        return 0
    elif week>0:
        return np.ceil(week/4)
    elif week<0:
        return np.floor(week/4)
    
def run_regression_worker(rq, time_unit, agg, est, identity, tweet_type):
    assert rq in ['language','activity']
    assert time_unit in ['month','week']
    if rq=='language':
        assert agg in ['mean','max','count']
    elif rq=='activity':
        if agg!='count':
            print("if rq=='activity', agg can only be 'count'")
            return
        # assert agg=='count'
    assert est in ['abs','rel']
    assert tweet_type in ['tweet','retweet','all']
    
    # save_dir=f'/shared/3/projects/bio-change/data/interim/regressions/{rq}/results-{tweet_type}-{time_unit}-{agg}-{est}'
    # cov_file=f'/shared/3/projects/bio-change/data/interim/propensity-score-matching/all-matches/propensity/with_tweet_identity/all_covariates.{identity}.df.tsv'
    # tweet_dir='/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/tweets_by_identity'
    # score_dir='/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/tweets_by_identity_scores'
    
    save_dir=f'/scratch/drom_root/drom0/minje/bio-change/06.regression/save_dir/{rq}/results-{tweet_type}-{time_unit}-{agg}-{est}'
    cov_file=f'/scratch/drom_root/drom0/minje/bio-change/06.regression/cov_dir/all_covariates.{identity}.df.tsv'
    tweet_dir='/scratch/drom_root/drom0/minje/bio-change/06.regression/tweet_dir'
    score_dir='/scratch/drom_root/drom0/minje/bio-change/06.regression/score_dir'
    
    time_unit_col = f'{time_unit}_diff'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    for file in os.listdir(save_dir):
        if file.startswith(f'table.{identity}'):
            print(f'File {file} already exists! Skipping...')
            return

    # load covariates
    df_cov=pd.read_csv(cov_file,sep='\t',dtype={'user_id':str})

    # load all tweet info and match to score
    if tweet_type=='all':
        df_tweet=pd.read_csv(join(tweet_dir,f'activities_made.{identity}.tweet.df.tsv.gz'),sep='\t',dtype={'user_id':str})
        df_retweet=pd.read_csv(join(tweet_dir,f'activities_made.{identity}.retweet.df.tsv.gz'),sep='\t',dtype={'user_id':str})
        df_tweet=pd.concat([df_tweet,df_retweet],axis=0)            
    else:
        df_tweet=pd.read_csv(join(tweet_dir,f'activities_made.{identity}.{tweet_type}.df.tsv.gz'),sep='\t',dtype={'user_id':str})
    # optional-add month values if time unit is month
    if time_unit=='month':
        df_tweet['month_diff']=[int(week_diff_to_month_diff(x)) for x in df_tweet.week_diff]
    # optional-add scores if RQ is language
    if rq=='language':
        with open(join(score_dir,f'{tweet_type}.activities_made.{identity}.txt')) as f:
            scores=[float(x) for x in f.readlines()]
        df_tweet['score']=scores
    elif rq=='activity':
        df_tweet['score']=1
        
    
    # get placeholder for each user and each week difference
    df_time=df_tweet[[time_unit_col]].drop_duplicates().sort_values(by=[time_unit_col])
    df1 = df_cov.merge(df_time,how='cross')
    if time_unit=='month':
        df1=df1[(df1.month_diff>=-1)&(df1.month_diff<=3)]
    if time_unit=='week':
        df1=df1[(df1.week_diff>=-4)&(df1.week_diff<=12)]
    
    # optional-get counts of tweets per week to add as control
    if rq=='language':
        df_cnt=df_tweet.groupby(['user_id',time_unit_col]).count().reset_index()[['user_id',time_unit_col,'score']]
        df_cnt=df_cnt.rename(columns = {'score':'activity_count'})
        df2=df1.merge(df_cnt,on=['user_id',time_unit_col],how='left').fillna(0)
    elif rq=='activity':
        df2=df1

    # get outcome variable
    if rq=='language':
        if agg=='mean':
            df_score=df_tweet.groupby(['user_id',time_unit_col]).mean().reset_index()[['user_id',time_unit_col,'score']]
        elif agg=='max':
            df_score=df_tweet.groupby(['user_id',time_unit_col]).max().reset_index()[['user_id',time_unit_col,'score']]
        elif agg=='count':
            df_score = df_tweet[df_tweet.score>=0.5]
            df_score=df_score.groupby(['user_id',time_unit_col]).count().reset_index()[['user_id',time_unit_col,'score']]
    elif rq=='activity':
        df_score=df_tweet.groupby(['user_id',time_unit_col]).count().reset_index()[['user_id',time_unit_col,'score']]        
        
    # merge scores to existing table
    df3=df2.merge(df_score,on=['user_id',f'{time_unit}_diff'],how='left').fillna(0)

    # optional-change to log variables so that we can measure percentage change instead
    if est=='rel':
        # if outcome is score - omit missing values
        if agg in ['mean','max']:
            df3=df3[df3.score>0]
        # if outcome is count, add small value
        elif agg=='count':
            df3['score']+=0.1
        df3['score']=[np.log(x) for x in df3['score']]
    
    # remove time unit corresponding to zero - needed because we want to remove the week of treatment which may be volatile and doesn't have enough data when aggregated at monthly basis
    df3=df3[df3[f'{time_unit}_diff']!=0]
    
    # create additional column corresponding for (1) post-treatment time & (2) in treated group
    df3[f'{time_unit}_diff_treated']=[max(0,x) for x in df3[time_unit_col]] # all time units below 0 are set as zero
    df3[f'{time_unit}_diff_treated']=df3[f'{time_unit}_diff_treated']*df3['is_identity'] # all values<=0 now become 0 if they are assigned in control group
    
    valid_columns = [
        'fri','fol','sta', # user activity history
        'profile_score', # identity score of previous profile
        'n_days_since_profile', # number of days since account was created
        'is_identity', # is treated user
        f'{time_unit}_diff' # contains slope for post-treatment activities (positive slope means increasing trend, negative slope means decreasing trend)
        ]
    if rq=='language':
        valid_columns.append('activity_count') # add column for total activity count
    
    # set up a regression task using statsmodels
    scaler = StandardScaler()
    X = pd.concat(
        [
            df3[valid_columns],
            pd.get_dummies(df3['week_treated'],prefix='week_treated',drop_first=True), # week for when profile was updated
            pd.get_dummies(df3[f'{time_unit}_diff'],prefix=f'{time_unit}_diff',drop_first=True),
            # pd.get_dummies(df3[f'{time_unit}_diff_treated'],prefix=f'{time_unit}_diff_treated',drop_first=True)        
        ],
        axis=1
    )
    y = df3.score
    groups=df3.user_id
    X[['fri','fol','sta']]=scaler.fit_transform(X[['fri','fol','sta']])
    # optional-normalize the activity count if language&count
    if (rq=='language') & (est=='rel'):
        X['activity_count']=[np.log(x+0.1) for x in X['activity_count']]
    X = sm.add_constant(X)

    # run model    
    model=sm.MixedLM(endog=y, exog=X, groups=groups)
    result=model.fit()
    print(result.summary())
        
    # run reduced model
    drop_columns = ['is_identity']+[col for col in X.columns if f'{time_unit}_diff_treated' in col]
    X2 = X.drop(columns=drop_columns,axis=1)
    model2=sm.MixedLM(endog=y, exog=X2, groups=groups)
    result2=model2.fit()
    
    # perform log-likelihood test
    llf = result.llf
    llr = result2.llf
    LR_statistic = -2*(llr-llf)
    #calculate p-value of test statistic using 2 degrees of freedom
    p_val = chi2.sf(LR_statistic, len(drop_columns))
    p_val = round(p_val,3)

    # save results as dataframe
    res=result.conf_int()
    res.columns=['coef_low','coef_high']
    res['coef']=result.params # coef
    res['pval']=result.pvalues
    res = res.reset_index()
    save_file=join(save_dir,f'table.{identity}.chi2_{p_val}.df.tsv')
    res = res[['index','pval','coef','coef_low','coef_high']]
    res.to_csv(save_file,sep='\t',index=False)
    print(f"Finished saving results for post-profile language change in {tweet_type}:{identity}")
    return

def run_regression_past_worker(rq, time_unit, agg, est, identity, tweet_type):
    assert rq in ['language']
    assert time_unit in ['week']
    if rq=='language':
        assert agg in ['mean','max','count']
    elif rq=='activity':
        if agg!='count':
            print("if rq=='activity', agg can only be 'count'")
            return
        # assert agg=='count'
    assert est in ['abs','rel']
    assert tweet_type in ['tweet','retweet']
    
    # save_dir=f'/shared/3/projects/bio-change/data/interim/regressions/{rq}/results-{tweet_type}-{time_unit}-{agg}-{est}'
    # cov_file=f'/shared/3/projects/bio-change/data/interim/propensity-score-matching/all-matches/propensity/with_tweet_identity/all_covariates.{identity}.df.tsv'
    # tweet_dir='/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/tweets_by_identity'
    # score_dir='/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/tweets_by_identity_scores'
    
    save_dir=f'/scratch/drom_root/drom0/minje/bio-change/06.regression/identity_added-without_tweet_identity/save_dir/{rq}/results-{tweet_type}-{time_unit}-{agg}-{est}'
    tweet_dir='scratch/drom_root/drom0/minje/bio-change/06.regression/identity_added-without_tweet_identity/tweet_dir'
    score_dir='scratch/drom_root/drom0/minje/bio-change/06.regression/identity_added-without_tweet_identity/score_dir'
    cov_file=f'/scratch/drom_root/drom0/minje/bio-change/06.regression/identity_added-without_tweet_identity/cov_dir/all_covariates.{identity}.df.tsv'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    for file in os.listdir(save_dir):
        if file.startswith(f'table.{identity}'):
            print(f'File {file} already exists! Skipping...')
            return

    # load covariates
    df_cov=pd.read_csv(cov_file,sep='\t',dtype={'user_id':str})

    # load all tweet info and match to score
    df_tweet=pd.read_csv(join(tweet_dir,f'activities_made.{identity}.{tweet_type}.df.tsv.gz'),sep='\t',dtype={'user_id':str})
    # optional-add month values if time unit is month
    if time_unit=='month':
        df_tweet['month_diff']=[int(week_diff_to_month_diff(x)) for x in df_tweet.week_diff]
    time_unit_col = f'{time_unit}_diff'
    # optional-add scores if RQ is language
    if rq=='language':
        with open(join(score_dir,f'{tweet_type}.activities_made.{identity}.txt')) as f:
            scores=[float(x) for x in f.readlines()]
        df_tweet['score']=scores
    elif rq=='activity':
        df_tweet['score']=1
        
    
    # get placeholder for each user and each week difference
    df_time=df_tweet[[time_unit_col]].drop_duplicates().sort_values(by=[time_unit_col])
    df1 = df_cov.merge(df_time,how='cross')
    if time_unit=='month':
        df1=df1[(df1.month_diff>=-1)&(df1.month_diff<=0)]
    if time_unit=='week':
        df1=df1[(df1.week_diff>=-4)&(df1.week_diff<=0)]
    
    # optional-get counts of tweets per week to add as control
    if rq=='language':
        df_cnt=df_tweet.groupby(['user_id',time_unit_col]).count().reset_index()[['user_id',time_unit_col,'score']]
        df_cnt=df_cnt.rename(columns = {'score':'activity_count'})
        df2=df1.merge(df_cnt,on=['user_id',time_unit_col],how='left').fillna(0)
    elif rq=='activity':
        df2=df1

    # get outcome variable
    if rq=='language':
        if agg=='mean':
            df_score=df_tweet.groupby(['user_id',time_unit_col]).mean().reset_index()[['user_id',time_unit_col,'score']]
        elif agg=='max':
            df_score=df_tweet.groupby(['user_id',time_unit_col]).max().reset_index()[['user_id',time_unit_col,'score']]
        elif agg=='count':
            df_score = df_tweet[df_tweet.score>=0.5]
            df_score=df_score.groupby(['user_id',time_unit_col]).count().reset_index()[['user_id',time_unit_col,'score']]
    elif rq=='activity':
        df_score=df_tweet.groupby(['user_id',time_unit_col]).count().reset_index()[['user_id',time_unit_col,'score']]        
        
    # merge scores to existing table
    df3=df2.merge(df_score,on=['user_id',f'{time_unit}_diff'],how='left').fillna(0)

    # optional-change to log variables so that we can measure percentage change instead
    if est=='rel':
        # if outcome is score - omit missing values
        if agg in ['mean','max']:
            df3=df3[df3.score>0]
        # if outcome is count, add small value
        elif agg=='count':
            df3['score']+=0.1
        df3['score']=[np.log(x) for x in df3['score']]
    
    # # remove time unit corresponding to zero - needed because we want to remove the week of treatment which may be volatile and doesn't have enough data when aggregated at monthly basis
    # df3=df3[df3[f'{time_unit}_diff']!=0]
    
    # create additional column corresponding for (1) post-treatment time & (2) in treated group
    df3[f'{time_unit}_diff_treated']=[max(0,x) for x in df3[time_unit_col]] # all time units below 0 are set as zero
    df3[f'{time_unit}_diff_treated']=df3[f'{time_unit}_diff_treated']*df3['is_identity'] # all values<=0 now become 0 if they are assigned in control group
    
    valid_columns = [
        'fri','fol','sta', # user activity history
        'profile_score', # identity score of previous profile
        'n_days_since_profile', # number of days since account was created
        'is_identity', # is treated user
        f'{time_unit}_diff' # contains slope for post-treatment activities (positive slope means increasing trend, negative slope means decreasing trend)
        ]
    if rq=='language':
        valid_columns.append('activity_count') # add column for total activity count
    
    # set up a regression task using statsmodels
    scaler = StandardScaler()
    X = pd.concat(
        [
            df3[valid_columns],
            pd.get_dummies(df3['week_treated'],prefix='week_treated',drop_first=True), # week for when profile was updated
            pd.get_dummies(df3[f'{time_unit}_diff'],prefix=f'{time_unit}_diff',drop_first=True),
            # pd.get_dummies(df3[f'{time_unit}_diff_treated'],prefix=f'{time_unit}_diff_treated',drop_first=True)        
        ],
        axis=1
    )
    y = df3.score
    groups=df3.user_id
    X[['fri','fol','sta']]=scaler.fit_transform(X[['fri','fol','sta']])
    # optional-normalize the activity count if language&count
    if (rq=='language') & (est=='rel'):
        X['activity_count']=[np.log(x+0.1) for x in X['activity_count']]
    X = sm.add_constant(X)

    # run model    
    model=sm.MixedLM(endog=y, exog=X, groups=groups)
    result=model.fit()
    print(result.summary())
        
    # run reduced model
    drop_columns = ['is_identity']+[col for col in X.columns if f'{time_unit}_diff_treated' in col]
    X2 = X.drop(columns=drop_columns,axis=1)
    model2=sm.MixedLM(endog=y, exog=X2, groups=groups)
    result2=model2.fit()
    
    # perform log-likelihood test
    llf = result.llf
    llr = result2.llf
    LR_statistic = -2*(llr-llf)
    #calculate p-value of test statistic using 2 degrees of freedom
    p_val = chi2.sf(LR_statistic, len(drop_columns))
    p_val = round(p_val,3)

    # save results as dataframe
    res=result.conf_int()
    res.columns=['coef_low','coef_high']
    res['coef']=result.params # coef
    res['pval']=result.pvalues
    res = res.reset_index()
    save_file=join(save_dir,f'table.{identity}.chi2_{p_val}.df.tsv')
    res = res[['index','pval','coef','coef_low','coef_high']]
    res.to_csv(save_file,sep='\t',index=False)
    print(f"Finished saving results for post-profile language change in {tweet_type}:{identity}")
    return

def run_offensive_regression_worker(rq, time_unit, agg, est, identity, tweet_type):
    assert rq=='offensive'
    assert time_unit in ['month','week']
    assert agg in ['mean','max','count']
    assert est in ['abs','rel']
    assert tweet_type=='tweet'
    
    save_dir=f'/shared/3/projects/bio-change/data/interim/regressions/{rq}/results-{tweet_type}-{time_unit}-{agg}-{est}'
    cov_file=f'/shared/3/projects/bio-change/data/interim/propensity-score-matching/all-matches/propensity/with_tweet_identity/all_covariates.{identity}.df.tsv'
    tweet_dir='/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/tweets_by_identity'
    identity_score_dir='/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/tweets_by_identity_scores'
    offensive_score_dir='/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/offensiveness-scores'
    
    save_dir=f'/scratch/drom_root/drom0/minje/bio-change/06.regression/save_dir/{rq}/results-{tweet_type}-{time_unit}-{agg}-{est}'
    cov_file=f'/scratch/drom_root/drom0/minje/bio-change/06.regression/cov_dir/all_covariates.{identity}.df.tsv'
    tweet_dir='/scratch/drom_root/drom0/minje/bio-change/06.regression/tweet_dir'
    identity_score_dir='/scratch/drom_root/drom0/minje/bio-change/06.regression/score_dir'
    offensive_score_dir='/scratch/drom_root/drom0/minje/bio-change/06.regression/offensive_dir'
    
    time_unit_col = f'{time_unit}_diff'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    for file in os.listdir(save_dir):
        if file.startswith(f'table.{identity}'):
            print(f'File {file} already exists! Skipping...')
            return

    # load covariates
    df_cov=pd.read_csv(cov_file,sep='\t',dtype={'user_id':str})

    # load all tweet info and match to score
    df_tweet=pd.read_csv(join(tweet_dir,f'activities_origin.{identity}.{tweet_type}.df.tsv.gz'),sep='\t',dtype={'user_id':str})
    # optional-add month values if time unit is month
    if time_unit=='month':
        df_tweet['month_diff']=[int(week_diff_to_month_diff(x)) for x in df_tweet.week_diff]
        
    # get placeholder for each user and each week difference
    df_time=df_tweet[[time_unit_col]].drop_duplicates().sort_values(by=[time_unit_col])
    df1 = df_cov.merge(df_time,how='cross')
    if time_unit=='month':
        df1=df1[(df1.month_diff>=-1)&(df1.month_diff<=3)]
    if time_unit=='week':
        df1=df1[(df1.week_diff>=-4)&(df1.week_diff<=12)]
        
    # add offensiveness scores by others to dataframe    
    with open(join(offensive_score_dir,f'activities_origin.{identity}.{tweet_type}.txt')) as f:
        scores=[float(x) for x in f.readlines()]
    df_tweet['score']=scores        
    df_tweet=df_tweet[['user_id',time_unit_col,'score']]
    if agg=='mean':
        df_tweet=df_tweet.groupby(['user_id',time_unit_col]).mean().reset_index()
    elif agg=='max':
        df_tweet=df_tweet.groupby(['user_id',time_unit_col]).max().reset_index()
    if agg=='count':
        df_tweet=df_tweet[df_tweet.score>=0.5].groupby(['user_id',time_unit_col]).count().reset_index()
    df2=df1.merge(df_tweet,on=['user_id',time_unit_col],how='left').fillna(0)

    # get counts of tweets per week to add as control
    df_cnt=df_tweet.groupby(['user_id',time_unit_col]).count().reset_index()[['user_id',time_unit_col,'score']]
    df_cnt=df_cnt.rename(columns = {'score':'activity_count'})
    df3=df2.merge(df_cnt,on=['user_id',time_unit_col],how='left').fillna(0)
    
    # get offensiveness scores of tweets posted by the ego user
    df_tweet=pd.read_csv(join(tweet_dir,f'activities_made.{identity}.{tweet_type}.df.tsv.gz'),sep='\t',dtype={'user_id':str})
    if time_unit=='month':
        df_tweet['month_diff']=[int(week_diff_to_month_diff(x)) for x in df_tweet.week_diff]
    with open(join(offensive_score_dir,f'activities_made.{identity}.{tweet_type}.txt')) as f:
        scores=[float(x) for x in f.readlines()]
    df_tweet['offensive_ego_score']=scores
    df_tweet=df_tweet[['user_id',time_unit_col,'offensive_ego_score']]
    if agg=='mean':
        df_tweet=df_tweet.groupby(['user_id',time_unit_col]).mean().reset_index()
    elif agg=='max':
        df_tweet=df_tweet.groupby(['user_id',time_unit_col]).max().reset_index()
    if agg=='count':
        df_tweet=df_tweet[df_tweet.offensive_ego_score>=0.5].groupby(['user_id',time_unit_col]).count().reset_index()
    df4=df3.merge(df_tweet,on=['user_id',time_unit_col],how='left').fillna(0)
    
    # get identity scores of tweets posted by the ego user
    df_tweet=pd.read_csv(join(tweet_dir,f'activities_made.{identity}.{tweet_type}.df.tsv.gz'),sep='\t',dtype={'user_id':str})
    if time_unit=='month':
        df_tweet['month_diff']=[int(week_diff_to_month_diff(x)) for x in df_tweet.week_diff]
    with open(join(identity_score_dir,f'{tweet_type}.activities_made.{identity}.txt')) as f:
        scores=[float(x) for x in f.readlines()]
    df_tweet['identity_ego_score']=scores
    df_tweet=df_tweet[['user_id',time_unit_col,'identity_ego_score']]
    if agg=='mean':
        df_tweet=df_tweet.groupby(['user_id',time_unit_col]).mean().reset_index()
    elif agg=='mean':
        df_tweet=df_tweet.groupby(['user_id',time_unit_col]).max().reset_index()
    if agg=='mean':
        df_tweet=df_tweet[df_tweet.identity_ego_score>=0.5].groupby(['user_id',time_unit_col]).count().reset_index()
    df5=df4.merge(df_tweet,on=['user_id',time_unit_col],how='left').fillna(0)
    

    # optional-change to log variables so that we can measure percentage change instead
    if est=='rel':
        # if outcome is score - omit missing values
        if agg in ['mean','max']:
            df5=df5[df5.score>0]
        # if outcome is count, add small value
        elif agg=='count':
            df5['score']+=0.1
        df5['score']=[np.log(x) for x in df5['score']]
    
    # remove time unit corresponding to zero - needed because we want to remove the week of treatment which may be volatile and doesn't have enough data when aggregated at monthly basis
    df5=df5[df5[f'{time_unit}_diff']!=0]
    
    # create additional column corresponding for (1) post-treatment time & (2) in treated group
    df5[f'{time_unit}_diff_treated']=[max(0,x) for x in df5[time_unit_col]] # all time units below 0 are set as zero
    df5[f'{time_unit}_diff_treated']=df5[f'{time_unit}_diff_treated']*df5['is_identity'] # all values<=0 now become 0 if they are assigned in control group
    
    valid_columns = [
        'fri','fol','sta', # user activity history
        'profile_score', # identity score of previous profile
        'n_days_since_profile', # number of days since account was created
        'is_identity', # is treated user
        'offensive_ego_score',
        'identity_ego_score',
        'activity_count',
        f'{time_unit}_diff' # this contains the slope value
        ]
    
    # set up a regression task using statsmodels
    scaler = StandardScaler()
    X = pd.concat(
        [
            df5[valid_columns],
            pd.get_dummies(df5['week_treated'],prefix='week_treated',drop_first=True), # week for when profile was updated
            pd.get_dummies(df5[f'{time_unit}_diff'],prefix=f'{time_unit}_diff',drop_first=True),
            # pd.get_dummies(df5[f'{time_unit}_diff_treated'],prefix=f'{time_unit}_diff_treated',drop_first=True)        
        ],
        axis=1
    )
    y = df5.score
    groups=df5.user_id
    X[['fri','fol','sta']]=scaler.fit_transform(X[['fri','fol','sta']])
    # optional-normalize the activity count if language&count
    if (est=='rel'):
        X['activity_count']=[np.log(x+0.1) for x in X['activity_count']]
    X = sm.add_constant(X)

    # run model    
    model=sm.MixedLM(endog=y, exog=X, groups=groups)
    result=model.fit()
    print(result.summary())
        
    # run reduced model
    drop_columns = ['is_identity']+[col for col in X.columns if f'{time_unit}_diff_treated' in col]
    X2 = X.drop(columns=drop_columns,axis=1)
    model2=sm.MixedLM(endog=y, exog=X2, groups=groups)
    result2=model2.fit()
    
    # perform log-likelihood test
    llf = result.llf
    llr = result2.llf
    LR_statistic = -2*(llr-llf)
    #calculate p-value of test statistic using 2 degrees of freedom
    p_val = chi2.sf(LR_statistic, len(drop_columns))
    p_val = round(p_val,3)

    # save results as dataframe
    res=result.conf_int()
    res.columns=['coef_low','coef_high']
    res['coef']=result.params # coef
    res['pval']=result.pvalues
    res = res.reset_index()
    save_file=join(save_dir,f'table.{identity}.chi2_{p_val}.df.tsv')
    res = res[['index','pval','coef','coef_low','coef_high']]
    res.to_csv(save_file,sep='\t',index=False)
    print(f"Finished saving results for offensive language change in {tweet_type}:{identity}")
    return

def run_regression(idx=None):
    identities = get_identities()
    settings=[]
    inputs = []
    for est in ['rel','abs']:
        for time_unit in ['month','week']:
            for rq in ['language','activity']:
                settings.append((est,time_unit,rq))
    if idx:
        settings=[settings[int(idx)]]

    print('settings:',settings)
    for est,time_unit,rq in settings:
        for agg in ['count','mean','max']:
                for tweet_type in ['tweet','retweet']:
                    for identity in identities:
                        inputs.append((rq, time_unit, agg, est, identity, tweet_type))
                    
    for input in inputs:
        run_regression_worker(*input)
    return

def run_past_regression(idx=None):
    identities = get_identities()
    settings=[]
    inputs = []
    for est in ['rel','abs']:
        for time_unit in ['week']:
            for rq in ['language']:
                settings.append((est,time_unit,rq))
    if idx:
        settings=[settings[int(idx)]]

    print('settings:',settings)
    for est,time_unit,rq in settings:
        for agg in ['count','mean']:
                for tweet_type in ['tweet','retweet']:
                    for identity in identities:
                        inputs.append((rq, time_unit, agg, est, identity, tweet_type))
                    
    for input in inputs:
        run_past_regression_worker(*input)
    return

def run_offensive_regression(idx=None):
    identities = get_identities()
    settings = []
    rq='offensive'
    tweet_type='tweet'
    for est in ['rel','abs']:
        for time_unit in ['month','week']:
            settings.append((rq, time_unit, est, tweet_type))
                
    if idx:
        settings=[settings[int(idx)]]
    # rq, time_unit, agg, est, identity, tweet_type
    
    inputs = []
    for rq, time_unit, est, tweet_type in settings:
        for agg in ['count','mean','max']:
            for identity in identities:
                inputs.append((rq, time_unit, agg, est, identity, tweet_type))
                
    for input in inputs:
        run_offensive_regression_worker(*input)
    return


if __name__=='__main__':
    if len(sys.argv)>1:
        run_past_regression(sys.argv[1])
    else:
        run_past_regression()
