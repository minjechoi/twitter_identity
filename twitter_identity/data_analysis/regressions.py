import os
from os.path import join
from multiprocessing import Pool

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2

from twitter_identity.utils.utils import week_diff_to_month_diff, get_identities

def run_regression_language_change_worker(time_unit, agg, est, identity, tweet_type):
    save_dir=f'/shared/3/projects/bio-change/data/interim/rq-language-change-post-identity/results-{tweet_type}-{time_unit}-{agg}-{est}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    for file in os.listdir(save_dir):
        if file.startswith(f'table.{identity}'):
            print(f'File {file} already exists! Skipping...')
            return

    # if time_period=='pre':
    #     df_cov=pd.read_csv(f'/shared/3/projects/bio-change/data/interim/propensity-score-matching/all-matches/propensity/without_tweet_identity/all_covariates.{identity}.df.tsv',
    #                 sep='\t',dtype={'user_id':str})
    df_cov=pd.read_csv(f'/shared/3/projects/bio-change/data/interim/propensity-score-matching/all-matches/propensity/with_tweet_identity/all_covariates.{identity}.df.tsv',
                sep='\t',dtype={'user_id':str})

    # load all tweet info and match to score
    tweet_dir='/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/tweets_by_identity'
    df_tweet=pd.read_csv(join(tweet_dir,f'activities_made.{identity}.{tweet_type}.df.tsv.gz'),sep='\t',dtype={'user_id':str})
    df_tweet['month_diff']=[int(week_diff_to_month_diff(x)) for x in df_tweet.week_diff]
    score_dir='/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/tweets_by_identity_scores'
    with open(join(score_dir,f'{tweet_type}.activities_made.{identity}.txt')) as f:
        scores=[float(x) for x in f.readlines()]
    df_tweet['score']=scores
    if tweet_type=='tweet':
        df_tweet=df_tweet[df_tweet.tweet_type=='tweet']
    
    # get placeholder for each user and each week difference
    df_time=df_tweet[[f'{time_unit}_diff']].drop_duplicates().sort_values(by=[f'{time_unit}_diff'])
    df1 = df_cov.merge(df_time,how='cross')
    if time_unit=='month':
        df1=df1[(df1.month_diff>=-1)&(df1.month_diff<=3)]
    if time_unit=='week':
        df1=df1[(df1.week_diff>=-4)&(df1.week_diff<=12)]
    
    # get counts of tweets per week to add as control
    df_cnt=df_tweet.groupby(['user_id',f'{time_unit}_diff']).count().reset_index()[['user_id',f'{time_unit}_diff','score']]
    df_cnt=df_cnt.rename(columns = {'score':'activity_count'})
    df2=df1.merge(df_cnt,on=['user_id',f'{time_unit}_diff'],how='left').fillna(0)

    # get outcome variable
    if agg=='mean':
        df_score=df_tweet.groupby(['user_id',f'{time_unit}_diff']).mean().reset_index()[['user_id',f'{time_unit}_diff','score']]
    elif agg=='max':
        df_score=df_tweet.groupby(['user_id',f'{time_unit}_diff']).max().reset_index()[['user_id',f'{time_unit}_diff','score']]
    elif agg=='count':
        df_score = df_tweet[df_tweet.score>=0.5]
        df_score=df_score.groupby(['user_id',f'{time_unit}_diff']).count().reset_index()[['user_id',f'{time_unit}_diff','score']]
        
    # merge scores to existing table
    df3=df2.merge(df_score,on=['user_id',f'{time_unit}_diff'],how='left').fillna(0)
    
    if est=='rel':
        # change to log variables so that we can look into percentage change instead
        if agg in ['mean','max']:
            df3=df3[df3.score>0]
        elif agg=='count':
            df3['score']=[max(x,0.5) for x in df3['score']]
        df3['score']=[np.log(x) for x in df3['score']]
    
    # remove time unit corresponding to zero - needed because we want to focus more on effects after stabilization
    df3=df3[df3[f'{time_unit}_diff']!=0]
    
    # create additional column corresponding for (1) post-treatment time & (2) in treated group
    df3[f'{time_unit}_diff_treated']=[max(0,x) for x in df3[f'{time_unit}_diff']] # all time units below 0 are set as zero
    df3[f'{time_unit}_diff_treated']=df3[f'{time_unit}_diff_treated']*df3['is_identity'] # all values<=0 now become 0 if they are assigned in control group
    
    
    # set up a regression task using statsmodels
    scaler = StandardScaler()
    X = pd.concat(
        [
            df3[['fri','fol','sta','profile_score','n_days_since_profile','is_identity','activity_count'
            # 'activity_-4','activity_-3','activity_-2','activity_-1'
            ]],
            pd.get_dummies(df3['week_treated'],prefix='week_treated',drop_first=True),
            pd.get_dummies(df3[f'{time_unit}_diff'],prefix=f'{time_unit}_diff',drop_first=True),
            pd.get_dummies(df3[f'{time_unit}_diff_treated'],prefix=f'{time_unit}_diff_treated',drop_first=True)        
        ],
        axis=1
    )
    y = df3.score
    groups=df3.user_id
    X[['fri','fol','sta']]=scaler.fit_transform(X[['fri','fol','sta']])
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
    res.to_csv(save_file,sep='\t',index=False)
    print(f"Finished saving results for post-profile language change in {tweet_type}:{identity}")
    return

def run_regression_activity_change_worker(time_period, identity, tweet_type):
    if time_period=='pre':
        df_cov=pd.read_csv(f'/shared/3/projects/bio-change/data/interim/propensity-score-matching/all-matches/propensity/without_tweet_identity/all_covariates.{identity}.df.tsv',
                    sep='\t',dtype={'user_id':str})
    elif time_period=='post':
        df_cov=pd.read_csv(f'/shared/3/projects/bio-change/data/interim/propensity-score-matching/all-matches/propensity/with_tweet_identity/all_covariates.{identity}.df.tsv',
                    sep='\t',dtype={'user_id':str})

    # load all tweet info and aggregate to counts
    tweet_dir='/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/tweets_by_identity'
    if tweet_type in ['tweet','retweet']:
        df_tweet=pd.read_csv(join(tweet_dir,f'activities_made.{identity}.{tweet_type}.df.tsv.gz'),sep='\t',dtype={'user_id':str})
        df_tweet['month_diff']=[int(week_diff_to_month_diff(x)) for x in df_tweet.week_diff]
        if tweet_type=='tweet':
            df_tweet=df_tweet[df_tweet.tweet_type=='tweet']
    elif tweet_type=='all':
        df_tweet=pd.read_csv(join(tweet_dir,f'activities_made.{identity}.tweet.df.tsv.gz'),sep='\t',dtype={'user_id':str})
        df_retweet=pd.read_csv(join(tweet_dir,f'activities_made.{identity}.retweet.df.tsv.gz'),sep='\t',dtype={'user_id':str})
        df_tweet=pd.concat([df_tweet,df_retweet],axis=0)
        df_tweet['month_diff']=[int(week_diff_to_month_diff(x)) for x in df_tweet.week_diff]        
        
    # get placeholder for each user and each week difference
    df_week=df_tweet[['week_diff','month_diff']].drop_duplicates().sort_values(by=['week_diff'])
    df1 = df_cov.merge(df_week,how='cross')
    # remove range out of +-3 months
    df1=df1[(df1.week_diff>=-4)&(df1.week_diff<=12)]
    
    # get counts of tweets
    df2=df_tweet.groupby(['user_id','week_diff','month_diff']).count().reset_index()[['user_id','week_diff','month_diff','text']]
    df2.columns = ['user_id','week_diff','month_diff','tweet_count']
    df2=df1.merge(df2,on=['user_id','week_diff','month_diff'],how='left').fillna(0)
    
    # preprocess columns for regression
    df2=df2.rename(columns = {f'week_diff_{i}':f'activity_{i}' for i in range(-4,0)})

    # create additional column that contains the week only if (1) in treatment group and (2) at the time treatment was assigned
    df2['week_diff_treated']=[max(0,x+1) for x in df2.week_diff] # week 0 becomes +1, and weekd<=0 become 0
    df2['week_diff_treated']=df2['week_diff_treated']*df2['is_identity'] # all values<=0 now become 0 if they are assigned in control group
    df2['week_diff_treated']-=1 # revert back so that the unique values are -1 (all cases to disregard), 0 (week 0 of treated), 1 (week 1 of treated), ...
    
    # set up a regression task using statsmodels
    scaler = StandardScaler()
    X = pd.concat(
        [
            df2[['fri','fol','sta','profile_score','n_days_since_profile','is_identity',
            'activity_-4','activity_-3','activity_-2','activity_-1']],
            pd.get_dummies(df2.week_diff,prefix='week_diff',drop_first=True),
            pd.get_dummies(df2.week_treated,prefix='week_treated',drop_first=True),
            pd.get_dummies(df2.week_diff_treated,prefix='week_diff_treated',drop_first=True)        
        ],
        axis=1
    )
    y = df2.tweet_count
    groups=df2.user_id
    X[['fri','fol','sta']]=scaler.fit_transform(X[['fri','fol','sta']])
    X = sm.add_constant(X)

    # run model    
    model=sm.MixedLM(endog=y, exog=X, groups=groups)
    result=model.fit()
    print(result.summary())
        
    # run reduced model
    drop_columns = ['is_identity']+[col for col in X.columns if 'week_diff_treated' in col]
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
    res=res.reset_index()
    res.to_csv(f'/shared/3/projects/bio-change/data/interim/rq-activity-change-post-identity/results-count/table.{tweet_type}.{identity}.chi2_{p_val}.df.tsv',sep='\t',index=False)
    print(f"Finished saving results for post-profile language change in {tweet_type}:{identity}")

    return

def run_regression_offensive_change_worker(time_unit, identity, tweet_type):
    setting='with_tweet_identity'
    # get treated / control users
    df_cov=pd.read_csv(f'/shared/3/projects/bio-change/data/interim/propensity-score-matching/all-matches/propensity/{setting}/all_covariates.{identity}.df.tsv',
                    sep='\t',dtype={'user_id':str})
    
    # load offensiveness scores for tweets
    activity_type='activities_origin'
    tweet_dir='/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/tweets_by_identity'
    df_tweet=pd.read_csv(join(tweet_dir,f'{activity_type}.{identity}.{tweet_type}.df.tsv.gz'),sep='\t',dtype={'user_id':str})
    df_tweet['month_diff']=[int(week_diff_to_month_diff(x)) for x in df_tweet.week_diff]
    score_dir='/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/offensiveness-scores/'
    with open(join(score_dir,f'{activity_type}.{identity}.{tweet_type}.txt')) as f:
        scores=[float(x) for x in f.readlines()]
    df_tweet['offensive_score']=scores
    
    # get placeholder for each user and each week difference
    df_week=df_tweet[['week_diff','month_diff']].drop_duplicates().sort_values(by=['week_diff'])
    df1 = df_cov.merge(df_week,how='cross')
    df1=df1[(df1.week_diff>=-4)&(df1.week_diff<=12)]

    # merge with offensiveness of replies
    df2=df_tweet[['user_id','week_diff','offensive_score']].groupby(['user_id','week_diff']).max().reset_index()
    df2=df1.merge(df2,on=['user_id','week_diff'],how='left').fillna(0)

    # load identity scores of past tweets
    activity_type='activities_made'
    tweet_dir='/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/tweets_by_identity'
    df_tweet=pd.read_csv(join(tweet_dir,f'{activity_type}.{identity}.{tweet_type}.df.tsv.gz'),sep='\t',dtype={'user_id':str})
    df_tweet['month_diff']=[int(week_diff_to_month_diff(x)) for x in df_tweet.week_diff]
    score_dir='/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/tweets_by_identity_scores/'
    with open(join(score_dir,f'{tweet_type}.{activity_type}.{identity}.txt')) as f:
        scores=[float(x) for x in f.readlines()]
    df_tweet['tweet_score']=scores

    # merge with scores of tweets of user
    df3=df_tweet[['user_id','week_diff','tweet_score']].groupby(['user_id','week_diff']).max().reset_index()
    df2=df2.merge(df3,on=['user_id','week_diff'],how='left').fillna(0)


    return

def run_regression():
    identities = get_identities()
    inputs = []
    for identity in identities:
        # for time_unit in ['month','week']:
        for time_unit in ['month']:
            # for tweet_type in ['tweet','retweet']:
            for tweet_type in ['retweet']:
                for est in ['rel']:
                    for agg in ['mean']:
                    # for agg in ['count','mean','max']:
                        inputs.append((time_unit, agg, est, identity, tweet_type))
            # time_unit, agg, identity, tweet_type
            
    pool = Pool(12)
    # pool.starmap(run_regression_activity_change_worker, inputs)
    for input in inputs:
        # run_regression_activity_change_worker(*input)
        run_regression_language_change_worker(*input)
    # pool.starmap(run_regression_language_change_worker, inputs)

    # inputs=[('post','gender_nonbinary','retweet')]
    # run_regression_language_change_worker(*inputs[0])
    return


if __name__=='__main__':
    run_regression(time_period='month')
