import os
from os.path import join
import sys
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2
from statsmodels.discrete.count_model import ZeroInflatedPoisson,ZeroInflatedGeneralizedPoisson,Poisson,ZeroInflatedNegativeBinomialP
from statsmodels.genmod.generalized_estimating_equations import GEE
from twitter_identity.utils.utils import get_identities
import matplotlib.pyplot as plt
import seaborn as sns

from twitter_identity.utils.utils import get_identities

# added due to greatlakes
# def get_identities():
#     """Simply returns the list of identities
#     """
#     user_data_dir='/scratch/drom_root/drom0/minje/bio-change/06.regression/identity_added-without_tweet_identity/cov_dir'
#     identities = sorted([file.split('.')[1] for file in os.listdir(user_data_dir) if file.startswith('all_covariates')])
#     return identities
pid=os.getpid()

# 05.08.23
def regression_total_count(tweet_type='tweet',mode='added'):
    """
    Regression that looks at whether the number of tweets/retweets increased 
    """
    assert tweet_type in ['tweet','retweet'],"'tweet_type' should be either 'tweet' or 'retweet'!"
    assert mode in ['added','removed'],"'mode' should be either 'added' or 'removed'!"
    
    identities = get_identities()
    for identity in identities:
        score_dir='/shared/3/projects/bio-change/data/interim/rq-language-identity/scores'
        if mode=='added':
            cov_file=f'/shared/3/projects/bio-change/data/interim/propensity-score-matching/users/after-matching/change_added-with_text/all_covariates.{identity}.df.tsv'
        elif mode=='removed':
            cov_file=f'/shared/3/projects/bio-change/data/interim/propensity-score-matching/users/after-matching/change_removed-with_text/all_covariates.{identity}.df.tsv'
        
        # load covariates
        df_cov=pd.read_csv(cov_file,sep='\t',dtype={'user_id':str})
        df_cov=df_cov.rename(columns={'label':'in_treatment_group'})
        df_cov=df_cov[['user_id','week_treated','in_treatment_group','fri','fol','sta','profile_identity_score']]
        df0=pd.DataFrame([0,1],columns=['t>=T'])
        df_cov=df_cov.merge(df0,how='cross')
        
        # get tweet counts
        tweet_dir='/shared/3/projects/bio-change/data/interim/treated-matched-tweets/tweets-around-update'
        df_tweet=pd.read_csv(join(tweet_dir,f'{tweet_type}.{identity}.df.tsv.gz'),sep='\t',dtype={'user_id':str})
        df_tweet=df_tweet[(df_tweet.week_difference>=-4)&(df_tweet.week_difference<=4)]
        df_tweet['t>=T']=(df_tweet.week_difference>=0).astype(int)
        df1=df_tweet[['user_id','t>=T','text']]
        df1=df1.groupby(['user_id','t>=T']).count().reset_index()
        df1=df1.rename(columns={'text':f'n_total_{tweet_type}s'})
        df_cov=df_cov.merge(df1,on=['user_id','t>=T'],how='left').fillna(0)
        
        df_out = df_cov
        # process covariates
        df_out['treatment_effect']=df_out.in_treatment_group * df_out['t>=T']
        valid_columns = [
            'fri','fol','sta', # user activity history
            'profile_identity_score', # identity score of previous profile
            'in_treatment_group', # is assigned into treatment group
            'treatment_effect', # has been treated
            't>=T',
            ]

        # set up a regression task using statsmodels
        scaler = StandardScaler()
        X = pd.concat(
            [
                df_out[valid_columns],
                pd.get_dummies(df_out['week_treated'],prefix='week_treated',drop_first=True),
            ],
            axis=1
        )
        X[['fri','fol','sta']]=scaler.fit_transform(X[['fri','fol','sta']])
        y = df_out[f'n_total_{tweet_type}s']
        groups=df_out.user_id
        X = sm.add_constant(X)
        
        for model_type in ['gee_poisson','poisson','negative_binomial',
                           'gee_negative_binomial']:
            save_dir = f'/shared/3/projects/bio-change/results/experiments/activity-change/regressions/total_count/{tweet_type}-{mode}/{model_type}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_file=join(save_dir,f'summary.{identity}.df.tsv')
            if os.path.exists(join(save_dir,save_file)):
                continue
            try:
                print(identity,model_type)
                if model_type=='gee_poisson':
                    model = GEE(endog=y,exog=X,family=sm.families.Poisson(), groups=groups)
                    result=model.fit(maxiter=10000)
                elif model_type=='poisson':
                    model = sm.Poisson(endog=y,exog=X)
                    result=model.fit_regularized(maxiter=10000)
                elif model_type=='negative_binomial':
                    model = sm.NegativeBinomialP(endog=y,exog=X)
                    result=model.fit_regularized(maxiter=10000)
                elif model_type=='gee_negative_binomial':
                    model = GEE(endog=y,exog=X,family=sm.families.NegativeBinomial(), groups=groups)
                    result=model.fit(maxiter=10000)
                    
                with open(save_file,'w') as f:
                    f.write(str(result.summary2()))
            except:
                with open(join(save_dir,f'error.{identity}.txt'),'w') as f:
                    f.write("Cannot complete regression!\nSkipping...")
                continue
            
                
            # save results as dataframe
            res=result.conf_int()
            res.columns=['coef_low','coef_high']
            res['coef']=result.params # coef
            res['pval']=result.pvalues
            res = res.reset_index()
            save_file=join(save_dir,f'table.{identity}.df.tsv')
            res = res[['index','pval','coef','coef_low','coef_high']]
            res.to_csv(save_file,sep='\t',index=False)
            print(f"Finished saving results for total activity change in {tweet_type}:{identity}:{mode}")
    return

def regression_identity_count(tweet_type='tweet',mode='added'):
    """
    Regression that looks at whether the number of tweets/retweets increased 
    """
    assert tweet_type in ['tweet','retweet'],"'tweet_type' should be either 'tweet' or 'retweet'!"
    assert mode in ['added','removed'],"'mode' should be either 'added' or 'removed'!"
    
    identities = get_identities()
    for identity in identities:
        if mode=='added':
            cov_file=f'/shared/3/projects/bio-change/data/interim/propensity-score-matching/users/after-matching/change_added-with_text/all_covariates.{identity}.df.tsv'
        elif mode=='removed':
            cov_file=f'/shared/3/projects/bio-change/data/interim/propensity-score-matching/users/after-matching/change_removed-with_text/all_covariates.{identity}.df.tsv'
        
        # load covariates
        df_cov=pd.read_csv(cov_file,sep='\t',dtype={'user_id':str})
        df_cov=df_cov.rename(columns={'label':'in_treatment_group'})
        df_cov=df_cov[['user_id','week_treated','in_treatment_group','fri','fol','sta','profile_identity_score']]
        df0=pd.DataFrame([0,1],columns=['t>=T'])
        df_cov=df_cov.merge(df0,how='cross')
        
        # get tweet counts
        tweet_dir='/shared/3/projects/bio-change/data/interim/treated-matched-tweets/tweets-around-update'
        df_tweet=pd.read_csv(join(tweet_dir,f'{tweet_type}.{identity}.df.tsv.gz'),sep='\t',dtype={'user_id':str})
        df_tweet=df_tweet[(df_tweet.week_difference>=-4)&(df_tweet.week_difference<=4)]
        df_tweet['t>=T']=(df_tweet.week_difference>=0).astype(int)
        df1=df_tweet[['user_id','t>=T','text']]
        df1=df1.groupby(['user_id','t>=T']).count().reset_index()
        df1=df1.rename(columns={'text':f'n_total_{tweet_type}s'})
        df_cov=df_cov.merge(df1,on=['user_id','t>=T'],how='left').fillna(0)
        
        # get tweet identity scores
        tweet_dir='/shared/3/projects/bio-change/data/interim/rq-language-identity/input-texts'
        score_dir='/shared/3/projects/bio-change/data/interim/rq-language-identity/identity-scores'
        df_tweet=pd.read_csv(join(tweet_dir,f'{tweet_type}.{identity}.df.tsv.gz'),sep='\t',dtype={'user_id':str})
        with open(join(score_dir,f'{tweet_type}-classifier.{identity}.txt')) as f:
            scores=f.readlines()
        scores=[float(x) for x in scores]
        df_tweet['score']=scores
        df_tweet['t>=T']=(df_tweet.week_difference>=0).astype(int)
        df2=df_tweet[['user_id','t>=T','score']]
        df2=df2[df2.score>=0.5].groupby(['user_id','t>=T']).count().reset_index()
        df2=df2.rename(columns={'score':f'n_identity_{tweet_type}s'})        
        df_cov=df_cov.merge(df2,on=['user_id','t>=T'],how='left').fillna(0)

        df_out = df_cov
        # process covariates
        df_out['treatment_effect']=df_out.in_treatment_group * df_out['t>=T']
        valid_columns = [
            'fri','fol','sta', # user activity history
            'profile_identity_score', # identity score of previous profile
            'in_treatment_group', # is assigned into treatment group
            'treatment_effect', # has been treated
            't>=T',
            f'n_total_{tweet_type}s'
            ]

        # set up a regression task using statsmodels
        scaler = StandardScaler()
        X = pd.concat(
            [
                df_out[valid_columns],
                pd.get_dummies(df_out['week_treated'],prefix='week_treated',drop_first=True),
            ],
            axis=1
        )
        X[['fri','fol','sta']]=scaler.fit_transform(X[['fri','fol','sta']])
        X[f'n_total_{tweet_type}s']=[np.log(x+1) for x in X[f'n_total_{tweet_type}s']]
        y = df_out[f'n_identity_{tweet_type}s']
        groups=df_out.user_id
        X = sm.add_constant(X)
        
        for model_type in ['gee_poisson','poisson','negative_binomial','gee_negative_binomial']:
            save_dir = f'/shared/3/projects/bio-change/results/experiments/activity-change/regressions/identity_count/{tweet_type}-{mode}/{model_type}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_file=join(save_dir,f'summary.{identity}.df.tsv')
            if os.path.exists(join(save_dir,save_file)):
                continue
            try:
                print(identity,model_type)
                if model_type=='gee_poisson':
                    model = GEE(endog=y,exog=X,family=sm.families.Poisson(), groups=groups)
                    result=model.fit(maxiter=10000)
                elif model_type=='poisson':
                    model = sm.Poisson(endog=y,exog=X)
                    result=model.fit_regularized(maxiter=10000)
                elif model_type=='negative_binomial':
                    model = sm.NegativeBinomialP(endog=y,exog=X)
                    result=model.fit_regularized(maxiter=10000)
                # elif model_type=='zero_inflated_negative_binomial':
                #     model = ZeroInflatedNegativeBinomialP(endog=y, exog=X)
                #     result=model.fit_regularized(maxiter=10000)
                # elif model_type=='zero_inflated_poisson':
                #     model = result=model.fit_regularized(maxiter=10000)
                elif model_type=='gee_negative_binomial':
                    model = GEE(endog=y,exog=X,family=sm.families.NegativeBinomial(), groups=groups)
                    result=model.fit(maxiter=10000)
                    
                with open(save_file,'w') as f:
                    f.write(str(result.summary2()))
            except:
                with open(join(save_dir,f'error.{identity}.txt'),'w') as f:
                    f.write("Cannot complete regression!\nSkipping...")
                continue
                
            # save results as dataframe
            res=result.conf_int()
            res.columns=['coef_low','coef_high']
            res['coef']=result.params # coef
            res['pval']=result.pvalues
            res = res.reset_index()
            save_file=join(save_dir,f'table.{identity}.df.tsv')
            res = res[['index','pval','coef','coef_low','coef_high']]
            res.to_csv(save_file,sep='\t',index=False)
            print(f"Finished saving results for total activity change in {tweet_type}:{identity}:{mode}")
    return

def regression_offensive_count(tweet_type='tweet',mode='added'):
    """
    Regression that looks at whether the number of tweets/retweets increased 
    """
    assert mode in ['added','removed'],"'mode' should be either 'added' or 'removed'!"
    if tweet_type!='tweet':
        print("tweet_type should always be 'tweet'!")
        return
    
    identities = get_identities()
    for identity in identities:
        if mode=='added':
            cov_file=f'/shared/3/projects/bio-change/data/interim/propensity-score-matching/users/after-matching/change_added-with_text/all_covariates.{identity}.df.tsv'
        elif mode=='removed':
            cov_file=f'/shared/3/projects/bio-change/data/interim/propensity-score-matching/users/after-matching/change_removed-with_text/all_covariates.{identity}.df.tsv'
        
        # load covariates
        df_cov=pd.read_csv(cov_file,sep='\t',dtype={'user_id':str})
        df_cov=df_cov.rename(columns={'label':'in_treatment_group'})
        df_cov=df_cov[['user_id','week_treated','in_treatment_group','fri','fol','sta','profile_identity_score']]
        df0=pd.DataFrame([0,1],columns=['t>=T'])
        df_cov=df_cov.merge(df0,how='cross')
        
        # get tweet counts
        tweet_dir='/shared/3/projects/bio-change/data/interim/treated-matched-tweets/tweets-around-update'
        df_tweet=pd.read_csv(join(tweet_dir,f'tweet.{identity}.df.tsv.gz'),sep='\t',dtype={'user_id':str})
        df_tweet=df_tweet[(df_tweet.week_difference>=-4)&(df_tweet.week_difference<=4)]
        df_tweet['t>=T']=(df_tweet.week_difference>=0).astype(int)
        df1=df_tweet[['user_id','t>=T','text']]
        df1=df1.groupby(['user_id','t>=T']).count().reset_index()
        df1=df1.rename(columns={'text':f'n_total_tweets'})
        df_cov=df_cov.merge(df1,on=['user_id','t>=T'],how='left').fillna(0)
        
        # get tweet identity scores
        tweet_dir='/shared/3/projects/bio-change/data/interim/rq-language-identity/input-texts'
        score_dir='/shared/3/projects/bio-change/data/interim/rq-language-identity/identity-scores'
        df_tweet=pd.read_csv(join(tweet_dir,f'tweet.{identity}.df.tsv.gz'),sep='\t',dtype={'user_id':str})
        with open(join(score_dir,f'tweet-classifier.{identity}.txt')) as f:
            scores=f.readlines()
        scores=[float(x) for x in scores]
        df_tweet['score']=scores
        df_tweet['t>=T']=(df_tweet.week_difference>=0).astype(int)
        df2=df_tweet[['user_id','t>=T','score']]
        df2=df2[df2.score>=0.5].groupby(['user_id','t>=T']).count().reset_index()
        df2=df2.rename(columns={'score':f'n_identity_tweets'})        
        df_cov=df_cov.merge(df2,on=['user_id','t>=T'],how='left').fillna(0)

        # get tweet offensiveness scores
        score_dir='/shared/3/projects/bio-change/data/interim/rq-language-identity/offensiveness-scores'
        with open(join(score_dir,f'tweets-{identity}.txt')) as f:
            scores=f.readlines()
        scores=[float(x) for x in scores]
        df_tweet['score']=scores
        df3=df_tweet[['user_id','t>=T','score']]
        df3=df3[df3.score>=0.5].groupby(['user_id','t>=T']).count().reset_index()
        df3=df3.rename(columns={'score':f'n_offensive_tweets'})        
        df_cov=df_cov.merge(df3,on=['user_id','t>=T'],how='left').fillna(0)

        # get reply counts
        tweet_dir='/shared/3/projects/bio-change/data/interim/treated-matched-tweets/replies-around-update'
        df_tweet=pd.read_csv(join(tweet_dir,f'responses.{identity}.df.tsv.gz'),sep='\t',dtype={'user_id':str})
        df_tweet=df_tweet[(df_tweet.week_difference>=-4)&(df_tweet.week_difference<=4)]
        df_tweet['t>=T']=(df_tweet.week_difference>=0).astype(int)
        df4=df_tweet[['user_id','t>=T','text']]
        df4=df4.groupby(['user_id','t>=T']).count().reset_index()
        df4=df4.rename(columns={'text':f'n_total_replies'})
        df_cov=df_cov.merge(df4,on=['user_id','t>=T'],how='left').fillna(0)

        # get reply offensiveness scores
        tweet_dir='/shared/3/projects/bio-change/data/interim/rq-language-identity/input-texts'
        df_tweet=pd.read_csv(join(tweet_dir,f'responses.{identity}.df.tsv.gz'),sep='\t',dtype={'user_id':str})
        score_dir='/shared/3/projects/bio-change/data/interim/rq-language-identity/offensiveness-scores'
        with open(join(score_dir,f'responses-{identity}.txt')) as f:
            scores=f.readlines()
        scores=[float(x) for x in scores]
        df_tweet['score']=scores
        df_tweet=df_tweet[(df_tweet.week_difference>=-4)&(df_tweet.week_difference<=4)]
        df_tweet['t>=T']=(df_tweet.week_difference>=0).astype(int)
        df5=df_tweet[['user_id','t>=T','score']]
        df5=df5[df5.score>=0.5].groupby(['user_id','t>=T']).count().reset_index()
        df5=df5.rename(columns={'score':f'n_offensive_replies'})        
        df_cov=df_cov.merge(df5,on=['user_id','t>=T'],how='left').fillna(0)

        df_out = df_cov
        # process covariates
        df_out['treatment_effect']=df_out.in_treatment_group * df_out['t>=T']
        valid_columns = [
            'fri','fol','sta', # user activity history
            'profile_identity_score', # identity score of previous profile
            'in_treatment_group', # is assigned into treatment group
            'treatment_effect', # has been treated
            't>=T',
            'n_total_tweets',
            'n_identity_tweets',
            'n_offensive_tweets',
            'n_total_replies',
            ]

        # set up a regression task using statsmodels
        scaler = StandardScaler()
        X = pd.concat(
            [
                df_out[valid_columns],
                pd.get_dummies(df_out['week_treated'],prefix='week_treated',drop_first=True),
            ],
            axis=1
        )
        X[['fri','fol','sta']]=scaler.fit_transform(X[['fri','fol','sta']])
        X['n_total_tweets']=[np.log(x+1) for x in X[f'n_total_tweets']]
        X['n_identity_tweets']=[np.log(x+1) for x in X[f'n_identity_tweets']]
        X['n_offensive_tweets']=[np.log(x+1) for x in X[f'n_offensive_tweets']]
        X['n_total_replies']=[np.log(x+1) for x in X[f'n_total_replies']]
        y = df_out['n_offensive_replies']
        groups=df_out.user_id
        X = sm.add_constant(X)
        
        for model_type in ['gee_poisson','poisson','negative_binomial',
                           'gee_negative_binomial']:
            save_dir = f'/shared/3/projects/bio-change/results/experiments/activity-change/regressions/offensive_count/{tweet_type}-{mode}/{model_type}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_file=join(save_dir,f'summary.{identity}.df.tsv')
            if os.path.exists(join(save_dir,save_file)):
                continue
            try:
                print(identity,model_type)
                if model_type=='gee_poisson':
                    model = GEE(endog=y,exog=X,family=sm.families.Poisson(), groups=groups)
                    result=model.fit(maxiter=10000)
                elif model_type=='poisson':
                    model = sm.Poisson(endog=y,exog=X)
                    result=model.fit_regularized(maxiter=10000)
                elif model_type=='negative_binomial':
                    model = sm.NegativeBinomialP(endog=y,exog=X)
                    result=model.fit_regularized(maxiter=10000)
                elif model_type=='gee_negative_binomial':
                    model = GEE(endog=y,exog=X,family=sm.families.NegativeBinomial(), groups=groups)
                    result=model.fit(maxiter=10000)
                    
                with open(save_file,'w') as f:
                    f.write(str(result.summary2()))
            except:
                with open(join(save_dir,f'error.{identity}.txt'),'w') as f:
                    f.write("Cannot complete regression!\nSkipping...")
                continue
                
            # save results as dataframe
            res=result.conf_int()
            res.columns=['coef_low','coef_high']
            res['coef']=result.params # coef
            res['pval']=result.pvalues
            res = res.reset_index()
            save_file=join(save_dir,f'table.{identity}.df.tsv')
            res = res[['index','pval','coef','coef_low','coef_high']]
            res.to_csv(save_file,sep='\t',index=False)
            print(f"Finished saving results for offensive count change in tweet:{identity}:{mode}")
    return



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
    
def run_regression_worker(rq, time_unit, agg, est, identity, tweet_type, model_type):
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
    assert model_type in ['poisson','gee','glm']
    
    # save_dir=f'/shared/3/projects/bio-change/data/processed/regressions/{rq}/{model_type}/data-{tweet_type}-{time_unit}-{agg}-{est}'
    save_dir=f'/shared/3/projects/bio-change/results/experiments/activity-change/regressions/{rq}/{model_type}/results-{tweet_type}-{time_unit}-{agg}-{est}'
    cov_file=f'/shared/3/projects/bio-change/data/interim/propensity-score-matching/all-matches/propensity/with_tweet_identity/all_covariates.{identity}.df.tsv'
    tweet_dir='/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/tweets_by_identity'
    score_dir='/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/tweets_by_identity_scores'

    time_unit_col = f'{time_unit}_diff'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    for file in os.listdir(save_dir):
        if file.startswith(f'table.{identity}'):
            print(f'File {file} already exists! Skipping...')
            return

    # load covariates
    df_cov=pd.read_csv(cov_file,sep='\t',dtype={'user_id':str})
    df_cov=df_cov.rename(columns={'is_identity':'in_treatment_group'})
    print(df_cov.shape,' users in total')

    # load all tweet info and match to score
    df_tweet=pd.read_csv(join(tweet_dir,f'activities_made.{identity}.{tweet_type}.df.tsv.gz'),
                         sep='\t',dtype={'user_id':str})
    # optional-add month values if time unit is month
    if time_unit=='month':
        df_tweet['month_diff']=[int(week_diff_to_month_diff(x)) for x in df_tweet.week_diff]
    # optional-add scores if RQ is language
    if rq=='language':
        with open(join(score_dir,f'activities_made.{identity}.{tweet_type}.txt')) as f:
            scores=[float(x) for x in f.readlines()]
        df_tweet['score']=scores
        df_tweet['text'] = df_tweet['text'].fillna('')
        df_tweet['text']=[re.sub(r'(URL|@username)','',x).strip() for x in df_tweet['text']]
        df_tweet.loc[(df_tweet['text']==''),'score']=0        
    elif rq=='activity':
        df_tweet['score']=1
    
    # get placeholder for each user and each week difference
    df_time=df_tweet[[time_unit_col]].drop_duplicates().sort_values(by=[time_unit_col])
    df1 = df_cov.merge(df_time,how='cross')
    # if time_unit=='month':
    #     df1=df1[(df1.month_diff>=-1)&(df1.month_diff<=3)]
    # if time_unit=='week':
    #     df1=df1[(df1.week_diff>=-4)&(df1.week_diff<=12)]
    if time_unit=='month':
        df1=df1[(df1.month_diff>=-1)&(df1.month_diff<=1)]
    if time_unit=='week':
        df1=df1[(df1.week_diff>=-4)&(df1.week_diff<=4)]
    
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
    df3=df2.merge(df_score,on=['user_id',time_unit_col],how='left').fillna(0)
    
    # remove time unit corresponding to zero - needed because we want to remove the week of treatment which may be volatile and doesn't have enough data when aggregated at monthly basis
    # df3=df3[df3[f'{time_unit}_diff']!=0]
        
    # create additional column corresponding for (1) post-treatment time & (2) in treated group
    
    df3['post_treatment']=(df3[time_unit_col]>=0).astype(int)
    df3['treatment_effect'] = df3['post_treatment'] * df3['in_treatment_group']

    # df3[f'{time_unit}s_since_treatment']=[max(0,x) for x in df3[time_unit_col]]
    # df3[f'{time_unit}s_since_treatment']=df3[f'{time_unit}s_since_treatment']*df3['in_treatment_group'] # all values<=1 now become 0 if they are assigned in control group (slope of treatment)
    # df3['treatment_effect']=(df3[f'{time_unit}s_since_treatment']>0).astype(int) # whether unit has been treated (intercept of treatment)
    
    # df3.to_csv(join(save_dir,f'{identity}.df.tsv'),sep='\t',index=False)
        
    valid_columns = [
        'fri','fol','sta', # user activity history
        'profile_score', # identity score of previous profile
        'n_days_since_profile', # number of days since account was created
        'in_treatment_group', # is assigned into treatment group
        # 'post_treatment', # time is past t0
        'treatment_effect', # has been treated
        'propensity_score',
        # f'{time_unit}s_since_treatment', # contains slope for post-treatment activities (positive slope means increasing trend, negative slope means decreasing trend)
        ]
    if rq=='language':
        valid_columns.append('activity_count') # add column for total activity count
    
    # # set up a regression task using statsmodels
    # scaler = StandardScaler()
    X = pd.concat(
        [
            df3[valid_columns],
            pd.get_dummies(df3['strata'],prefix='strata',drop_first=True), # week for when profile was updated
            pd.get_dummies(df3['week_treated'],prefix='week_treated',drop_first=True), # week for when profile was updated
            pd.get_dummies(df3[time_unit_col],prefix=time_unit_col,drop_first=True),
        ],
        axis=1
    )
    y = df3.score
    groups=df3.user_id
    # X[['fri','fol','sta']]=scaler.fit_transform(X[['fri','fol','sta']])
    # # optional-normalize the activity count if language&count
    # if (rq=='language'):
    #     X['activity_count']=[np.log(x+0.1) for x in X['activity_count']]
    X = sm.add_constant(X)
    print(f"Starting regression for identity language change in {tweet_type}:{identity}")
    print(X.shape,' total features')

    # run model 
    if agg=='count':
        if model_type=='poisson':
            model=Poisson(endog=y, exog=X)
        elif model_type=='gee':
            model=GEE(endog=y, exog=X, groups=groups, family=sm.families.Poisson())
        elif model_type=='glm':
            model = sm.GLM(endog=y, exog=X, family=sm.families.Poisson())
        try:
            result=model.fit(maxiter=10000)
        except:
            print("Model can't be learned!")
            return

    else:
        model=sm.MixedLM(endog=y, exog=X, groups=groups)
        result=model.fit_regularized(maxiter=10000)
    # print(result.summary2())
    # llr_pval=round(result.llr_pvalue,3)

    save_file=join(save_dir,f'summary.{identity}.df.tsv')
    with open(save_file,'w') as f:
        f.write(str(result.summary2()))
        
    # # run reduced model
    # drop_columns = ['is_identity','is_treated',f'{time_unit}s_since_treatment']
    # X2 = X.drop(columns=drop_columns,axis=1)
    # model2=sm.MixedLM(endog=y, exog=X2, groups=groups)
    # result2=model2.fit()
    
    # # perform log-likelihood test
    # llf = result.llf
    # llr = result2.llf
    # LR_statistic = -2*(llr-llf)
    # #calculate p-value of test statistic using 2 degrees of freedom
    # p_val = chi2.sf(LR_statistic, len(drop_columns))
    # p_val = round(p_val,3)

    # save results as dataframe
    res=result.conf_int()
    res.columns=['coef_low','coef_high']
    res['coef']=result.params # coef
    res['pval']=result.pvalues
    res = res.reset_index()
    save_file=join(save_dir,f'table.{identity}.df.tsv')
    # save_file=join(save_dir,f'table.{identity}.llr_{llr_pval}.chi2_{p_val}.df.tsv')
    res = res[['index','pval','coef','coef_low','coef_high']]
    res.to_csv(save_file,sep='\t',index=False)
    print(f"[{pid}] Finished saving results for identity language change in {tweet_type}:{identity}")
    
    # save figure
    # g=sns.pointplot(data=df3,x=time_unit_col,y='score',hue='in_treatment_group')
    # g.set_title(f"[{identity}] identity-specific {tweet_type}s per week")
    # g.set_xlabel("Week difference")
    # g.set_ylabel("Average number of {tweet_type}s")
    # save_file=join(save_dir,f'plot.{identity}.pdf')
    # plt.tight_layout()
    # plt.savefig(save_file,bbox_inches='tight')
    return

def run_regression_past_worker(rq, time_unit, agg, est, identity, tweet_type, model_type):
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
    assert model_type in ['poisson','gee','mixedlm']

    
    save_dir=f'/shared/3/projects/bio-change/results/experiments/activity-change/{model_type}/{rq}_past/results-{tweet_type}-{time_unit}-{agg}-{est}'
    cov_file=f'/shared/3/projects/bio-change/data/interim/propensity-score-matching/all-matches/propensity/without_tweet_identity/all_covariates.{identity}.df.tsv'
    tweet_dir='/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/tweets_by_identity'
    score_dir='/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/tweets_by_identity_scores'
    
    # save_dir=f'/scratch/drom_root/drom0/minje/bio-change/06.regression/identity_added-without_tweet_identity/save_dir/{rq}/results-{tweet_type}-{time_unit}-{agg}-{est}'
    # tweet_dir='/scratch/drom_root/drom0/minje/bio-change/06.regression/identity_added-without_tweet_identity/tweet_dir'
    # score_dir='/scratch/drom_root/drom0/minje/bio-change/06.regression/identity_added-without_tweet_identity/score_dir'
    # cov_file=f'/scratch/drom_root/drom0/minje/bio-change/06.regression/identity_added-without_tweet_identity/cov_dir/all_covariates.{identity}.df.tsv'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    for file in os.listdir(save_dir):
        if file.startswith(f'table.{identity}'):
            print(f'File {file} already exists! Skipping...')
            return

    # load covariates
    df_cov=pd.read_csv(cov_file,sep='\t',dtype={'user_id':str})
    df_cov=df_cov.rename(columns={'is_identity':'in_treatment_group'})

    # load all tweet info and match to score
    df_tweet=pd.read_csv(join(tweet_dir,f'activities_made.{identity}.{tweet_type}.df.tsv.gz'),sep='\t',dtype={'user_id':str})
    # optional-add month values if time unit is month
    if time_unit=='month':
        df_tweet['month_diff']=[int(week_diff_to_month_diff(x)) for x in df_tweet.week_diff]
    time_unit_col = f'{time_unit}_diff'
    # optional-add scores if RQ is language
    if rq=='language':
        with open(join(score_dir,f'activities_made.{identity}.{tweet_type}.txt')) as f:
            scores=[float(x) for x in f.readlines()]
        df_tweet['score']=scores
        df_tweet['text'] = df_tweet['text'].fillna('')
        df_tweet['text']=[re.sub(r'(URL|@username)','',x).strip() for x in df_tweet['text']]
        df_tweet.loc[(df_tweet['text']==''),'score']=0        
    elif rq=='activity':
        df_tweet['score']=1
    
    # get placeholder for each user and each week difference
    df_time=df_tweet[[time_unit_col]].drop_duplicates().sort_values(by=[time_unit_col])
    df1 = df_cov.merge(df_time,how='cross')
    # df1=df1[(df1.week_diff>=-4)&(df1.week_diff<=0)]
    df1=df1[(df1.week_diff>=-12)&(df1.week_diff<=0)]
    
    # optional-get counts of tweets per week to add as control
    assert rq=='language'
    df_cnt=df_tweet.groupby(['user_id',time_unit_col]).count().reset_index()[['user_id',time_unit_col,'score']]
    df_cnt=df_cnt.rename(columns = {'score':'activity_count'})
    df2=df1.merge(df_cnt,on=['user_id',time_unit_col],how='left').fillna(0)

    # get outcome variable
    if agg=='mean':
        df_score=df_tweet.groupby(['user_id',time_unit_col]).mean().reset_index()[['user_id',time_unit_col,'score']]
    elif agg=='max':
        df_score=df_tweet.groupby(['user_id',time_unit_col]).max().reset_index()[['user_id',time_unit_col,'score']]
    elif agg=='count':
        df_score = df_tweet[df_tweet.score>=0.5]
        df_score=df_score.groupby(['user_id',time_unit_col]).count().reset_index()[['user_id',time_unit_col,'score']]
        
    # merge scores to existing table
    df3=df2.merge(df_score,on=['user_id',time_unit_col],how='left').fillna(0)

    # optional-change to log variables so that we can measure percentage change instead
    # if est=='rel':
    #     # if outcome is score - omit missing values
    #     if agg in ['mean','max']:
    #         df3=df3[df3.score>0]
    #     # if outcome is count, add small value
    #     elif agg=='count':
    #         df3['score']+=0.1
    #     df3['score']=[np.log(x) for x in df3['score']]
    
    # remove time unit corresponding to zero - needed because we want to remove the week of treatment which may be volatile and doesn't have enough data when aggregated at monthly basis
    if time_unit=='week':
        df3=df3[df3[time_unit_col]!=0]
    
    # create additional column corresponding for (1) post-treatment time & (2) in treated group
    # df3[time_unit_col]+=4
    df3[f'{time_unit}s_since_treatment']=df3[time_unit_col]*df3['in_treatment_group'] # all values<=1 now become 0 if they are assigned in control group (slope of treatment)
    
    valid_columns = [
        'fri','fol','sta', # user activity history
        'profile_score', # identity score of previous profile
        'n_days_since_profile', # number of days since account was created
        'in_treatment_group', # is treated user
        f'{time_unit}s_since_treatment', # contains slope for post-treatment activities (positive slope means increasing trend, negative slope means decreasing trend)
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
    X = sm.add_constant(X)

    # run model 
    if agg=='count':
        if model_type=='poisson':
            model=Poisson(endog=y, exog=X)
        elif model_type=='gee':
            model=GEE(endog=y, exog=X, groups=groups, family=sm.families.Poisson())
        elif model_type=='mixedlm':
            model = sm.MixedLM(endog=y, exog=X, family=sm.families.Poisson(), groups=groups)
        result=model.fit(maxiter=10000)
    else:
        model=sm.MixedLM(endog=y, exog=X, groups=groups)
        result=model.fit_regularized(maxiter=10000)
    print(result.summary2())

    save_file=join(save_dir,f'summary.{identity}.df.tsv')
    with open(save_file,'w') as f:
        f.write(str(result.summary2()))
        
    # save results as dataframe
    res=result.conf_int()
    res.columns=['coef_low','coef_high']
    res['coef']=result.params # coef
    res['pval']=result.pvalues
    res = res.reset_index()
    save_file=join(save_dir,f'table.{identity}.df.tsv')
    # save_file=join(save_dir,f'table.{identity}.llr_{llr_pval}.chi2_{p_val}.df.tsv')
    res = res[['index','pval','coef','coef_low','coef_high']]
    res.to_csv(save_file,sep='\t',index=False)
    print(f"Finished saving results for past language change in {tweet_type}:{identity}")
    return

def run_weekly_regression_worker(rq, time_unit, week, agg, est, tweet_type, model_type):
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
    assert model_type in ['poisson','gee','glm']
    
    save_dir=f'/shared/3/projects/bio-change/results/experiments/activity-change/{model_type}/weekly_effect/{rq}/results-{tweet_type}-{time_unit}-{agg}-{est}'
    
    tweet_dir='/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/tweets_by_identity'
    score_dir='/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/tweets_by_identity_scores'

    time_unit_col = f'{time_unit}_diff'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for file in os.listdir(save_dir):
        if file.startswith('table.%02d.'%(week)):
            print(f'File {file} already exists! Skipping...')
            return

    df_all = pd.DataFrame()
    
    for identity in get_identities():
        cov_file=f'/shared/3/projects/bio-change/data/interim/propensity-score-matching/all-matches/propensity/with_tweet_identity/all_covariates.{identity}.df.tsv'
        # load covariates
        df_cov=pd.read_csv(cov_file,sep='\t',dtype={'user_id':str})
        df_cov=df_cov.rename(columns={'is_identity':'in_treatment_group'})
        
        df_cov = df_cov[df_cov.week_treated==week]

        # load all tweet info and match to score
        df_tweet=pd.read_csv(join(tweet_dir,f'activities_made.{identity}.{tweet_type}.df.tsv.gz'),
                            sep='\t',dtype={'user_id':str})
        # optional-add month values if time unit is month
        if time_unit=='month':
            df_tweet['month_diff']=[int(week_diff_to_month_diff(x)) for x in df_tweet.week_diff]
        # optional-add scores if RQ is language
        if rq=='language':
            with open(join(score_dir,f'activities_made.{identity}.{tweet_type}.txt')) as f:
                scores=[float(x) for x in f.readlines()]
            df_tweet['score']=scores
            df_tweet['text'] = df_tweet['text'].fillna('')
            df_tweet['text']=[re.sub(r'(URL|@username)','',x).strip() for x in df_tweet['text']]
            df_tweet.loc[(df_tweet['text']==''),'score']=0        
        elif rq=='activity':
            df_tweet['score']=1
        
        # get placeholder for each user and each week difference
        df_time=df_tweet[[time_unit_col]].drop_duplicates().sort_values(by=[time_unit_col])
        df1 = df_cov.merge(df_time,how='cross')
        # if time_unit=='month':
        #     df1=df1[(df1.month_diff>=-1)&(df1.month_diff<=3)]
        # if time_unit=='week':
        #     df1=df1[(df1.week_diff>=-4)&(df1.week_diff<=12)]
        if time_unit=='month':
            df1=df1[(df1.month_diff>=-3)&(df1.month_diff<=3)]
        if time_unit=='week':
            df1=df1[(df1.week_diff>=-12)&(df1.week_diff<=12)]
        
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
        df3['identity_type']=identity
        df_all=pd.concat([df_all,df3],axis=0)
        print(f"Added {identity}! {len(df_all)} lines")
        
    df3=df_all
    
    # remove time unit corresponding to zero - needed because we want to remove the week of treatment which may be volatile and doesn't have enough data when aggregated at monthly basis
    df3=df3[df3[f'{time_unit}_diff']!=0]
    
    # create additional column corresponding for (1) post-treatment time & (2) in treated group
    df3[f'{time_unit}s_since_treatment']=[max(0,x) for x in df3[time_unit_col]]
    df3[f'{time_unit}s_since_treatment']=df3[f'{time_unit}s_since_treatment']*df3['in_treatment_group'] # all values<=1 now become 0 if they are assigned in control group (slope of treatment)
    df3['treatment_effect']=(df3[f'{time_unit}s_since_treatment']>0).astype(int) # whether unit has been treated (intercept of treatment)
        
    valid_columns = [
        'fri','fol','sta', # user activity history
        'profile_score', # identity score of previous profile
        'n_days_since_profile', # number of days since account was created
        'in_treatment_group', # is assigned into treatment group
        'treatment_effect', # has been treated
        # 'propensity_score',
        f'{time_unit}s_since_treatment', # contains slope for post-treatment activities (positive slope means increasing trend, negative slope means decreasing trend)
        ]
    if rq=='language':
        valid_columns.append('activity_count') # add column for total activity count
    
    # set up a regression task using statsmodels
    scaler = StandardScaler()
    X = pd.concat(
        [
            df3[valid_columns],
            # pd.get_dummies(df3['strata'],prefix='strata',drop_first=True), # week for when profile was updated
            pd.get_dummies(df3['identity_type'],prefix='identity:',drop_first=True), # week for when profile was updated
            pd.get_dummies(df3[f'{time_unit}_diff'],prefix=f'{time_unit}_diff',drop_first=True),
        ],
        axis=1
    )
    y = df3.score
    groups=df3.user_id
    X[['fri','fol','sta']]=scaler.fit_transform(X[['fri','fol','sta']])
    # optional-normalize the activity count if language&count
    if (rq=='language'):
        X['activity_count']=[np.log(x+0.1) for x in X['activity_count']]
    X = sm.add_constant(X)
    print(f"Starting regression for identity language change in {tweet_type}:{identity}")

    # run model 
    if agg=='count':
        if model_type=='poisson':
            model=Poisson(endog=y, exog=X)
        elif model_type=='gee':
            model=GEE(endog=y, exog=X, groups=groups, family=sm.families.Poisson())
        elif model_type=='glm':
            model = sm.GLM(endog=y, exog=X, family=sm.families.Poisson())
        result=model.fit(maxiter=10000)

    else:
        model=sm.MixedLM(endog=y, exog=X, groups=groups)
        result=model.fit_regularized(maxiter=10000)
    print(result.summary2())
    # llr_pval=round(result.llr_pvalue,3)

    save_file=join(save_dir,'summary.%02d.df.tsv'%(week))
    with open(save_file,'w') as f:
        f.write(str(result.summary2()))

    # save results as dataframe
    res=result.conf_int()
    res.columns=['coef_low','coef_high']
    res['coef']=result.params # coef
    res['pval']=result.pvalues
    res = res.reset_index()
    save_file=join(save_dir,'table.%02d.df.tsv'%week)
    res = res[['index','pval','coef','coef_low','coef_high']]
    res.to_csv(save_file,sep='\t',index=False)
    print(f"[{pid}] Finished saving results for identity language change in {tweet_type}:{identity}")
    
    # save figure
    # g=sns.pointplot(data=df3,x=time_unit_col,y='score',hue='in_treatment_group')
    # g.set_title(f"[{identity}] identity-specific {tweet_type}s per week")
    # g.set_xlabel("Week difference")
    # g.set_ylabel("Average number of {tweet_type}s")
    # save_file=join(save_dir,f'plot.{identity}.pdf')
    # plt.tight_layout()
    # plt.savefig(save_file,bbox_inches='tight')
    return

def run_offensive_regression_worker(rq, time_unit, agg, est, identity, tweet_type, model_type):
    assert rq=='offensive'
    assert time_unit in ['month','week']
    assert agg in ['mean','max','count']
    assert est in ['abs','rel']
    assert tweet_type=='tweet'
    assert model_type in ['poisson','gee','mixedlm']

    save_dir=f'/shared/3/projects/bio-change/data/interim/regressions/{rq}/{model_type}/results-{tweet_type}-{time_unit}-{agg}-{est}'
    cov_file=f'/shared/3/projects/bio-change/data/interim/propensity-score-matching/all-matches/propensity/with_tweet_identity/all_covariates.{identity}.df.tsv'
    tweet_dir='/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/tweets_by_identity'
    identity_score_dir='/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/tweets_by_identity_scores'
    offensive_score_dir='/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/offensiveness-scores'
        
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
    df_tweet['text'] = df_tweet['text'].fillna('')
    df_tweet['text']=[re.sub(r'(URL|@username)','',x).strip() for x in df_tweet['text']]
    df_tweet.loc[(df_tweet['text']==''),'score']=0        
    df_tweet=df_tweet[df_tweet.is_english==True]
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
    df_tweet['text'] = df_tweet['text'].fillna('')
    df_tweet['text']=[re.sub(r'(URL|@username)','',x).strip() for x in df_tweet['text']]
    df_tweet.loc[(df_tweet['text']==''),'offensive_ego_score']=0
    df_tweet=df_tweet[df_tweet.is_english==True]
    df_tweet=df_tweet[['user_id',time_unit_col,'offensive_ego_score']]
    if agg=='mean':
        df_tweet=df_tweet.groupby(['user_id',time_unit_col]).mean().reset_index()
    elif agg=='max':
        df_tweet=df_tweet.groupby(['user_id',time_unit_col]).max().reset_index()
    if agg=='count':
        df_tweet=df_tweet[df_tweet['offensive_ego_score']>=0.5].groupby(['user_id',time_unit_col]).count().reset_index()
    df4=df3.merge(df_tweet,on=['user_id',time_unit_col],how='left').fillna(0)
    
    # get identity scores of tweets posted by the ego user
    df_tweet=pd.read_csv(join(tweet_dir,f'activities_made.{identity}.{tweet_type}.df.tsv.gz'),sep='\t',dtype={'user_id':str})
    if time_unit=='month':
        df_tweet['month_diff']=[int(week_diff_to_month_diff(x)) for x in df_tweet.week_diff]
    with open(join(identity_score_dir,f'activities_made.{identity}.{tweet_type}.txt')) as f:
        scores=[float(x) for x in f.readlines()]
    df_tweet['identity_ego_score']=scores
    df_tweet['text'] = df_tweet['text'].fillna('')
    df_tweet['text']=[re.sub(r'(URL|@username)','',x).strip() for x in df_tweet['text']]
    df_tweet.loc[(df_tweet['text']==''),'identity_ego_score']=0
    df_tweet=df_tweet[df_tweet.is_english==True]
    df_tweet=df_tweet[['user_id',time_unit_col,'identity_ego_score']]
    if agg=='mean':
        df_tweet=df_tweet.groupby(['user_id',time_unit_col]).mean().reset_index()
    elif agg=='mean':
        df_tweet=df_tweet.groupby(['user_id',time_unit_col]).max().reset_index()
    if agg=='mean':
        df_tweet=df_tweet[df_tweet.identity_ego_score>=0.5].groupby(['user_id',time_unit_col]).count().reset_index()
    df5=df4.merge(df_tweet,on=['user_id',time_unit_col],how='left').fillna(0)
    

    # optional-change to log variables so that we can measure percentage change instead
    # if est=='rel':
    #     # if outcome is score - omit missing values
    #     if agg in ['mean','max']:
    #         df5=df5[df5.score>0]
    #     # if outcome is count, add small value
    #     elif agg=='count':
    #         df5['score']+=0.1
    #     df5['score']=[np.log(x) for x in df5['score']]
    
    # remove time unit corresponding to zero - needed because we want to remove the week of treatment which may be volatile and doesn't have enough data when aggregated at monthly basis
    df5=df5[df5[f'{time_unit}_diff']!=0]
    
    # create additional column corresponding for (1) post-treatment time & (2) in treated group
    df5[f'{time_unit}s_since_treatment']=[max(0,x) for x in df5[time_unit_col]]
    df5[f'{time_unit}s_since_treatment']=df5[f'{time_unit}s_since_treatment']*df5['in_treatment_group'] # all values<=1 now become 0 if they are assigned in control group (slope of treatment)
    df5['treatment_effect']=(df5[f'{time_unit}s_since_treatment']>0).astype(int) # whether unit has been treated (intercept of treatment)
    
    valid_columns = [
        'fri','fol','sta', # user activity history
        'profile_score', # identity score of previous profile
        'n_days_since_profile', # number of days since account was created
        'in_treatment_group', # is treated user
        'offensive_ego_score',
        'identity_ego_score',
        'activity_count',
        'treatment_effect', # has been treated
        f'{time_unit}s_since_treatment', # contains slope for post-treatment activities (positive slope means increasing trend, negative slope means decreasing trend)
        ]
    
    # set up a regression task using statsmodels
    scaler = StandardScaler()
    X = pd.concat(
        [
            df5[valid_columns],
            # pd.get_dummies(df5['strata'],prefix='strata',drop_first=True), # strata
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
    if model_type=='poisson':
        model=Poisson(endog=y, exog=X)
    elif model_type=='gee':
        model = GEE(endog=y, exog=X, family=sm.families.Poisson(), groups=groups)
    elif model_type=='mixedlm':
        model=sm.MixedLM(endog=y, exog=X, family=sm.families.Poisson(), groups=groups)
    result=model.fit()
    print(result.summary2())
    save_file=join(save_dir,f'summary.{identity}.df.tsv')
    with open(save_file,'w') as f:
        f.write(str(result.summary2()))

    # save results as dataframe
    res=result.conf_int()
    res.columns=['coef_low','coef_high']
    res['coef']=result.params # coef
    res['pval']=result.pvalues
    res = res.reset_index()
    save_file=join(save_dir,f'table.{identity}.df.tsv')
    res = res[['index','pval','coef','coef_low','coef_high']]
    res.to_csv(save_file,sep='\t',index=False)
    print(f"Finished saving results for offensiveness change in {tweet_type}:{identity}")
        
    # # run reduced model
    # drop_columns = ['is_identity','is_treated',f'{time_unit}s_since_treatment']
    # X2 = X.drop(columns=drop_columns,axis=1)
    # model2=sm.MixedLM(endog=y, exog=X2, groups=groups)
    # result2=model2.fit()
    
    # # perform log-likelihood test
    # llf = result.llf
    # llr = result2.llf
    # LR_statistic = -2*(llr-llf)
    # #calculate p-value of test statistic using 2 degrees of freedom
    # p_val = chi2.sf(LR_statistic, len(drop_columns))
    # p_val = round(p_val,3)

    # # save results as dataframe
    # res=result.conf_int()
    # res.columns=['coef_low','coef_high']
    # res['coef']=result.params # coef
    # res['pval']=result.pvalues
    # res = res.reset_index()
    # save_file=join(save_dir,f'table.{identity}.chi2_{p_val}.df.tsv')
    # res = res[['index','pval','coef','coef_low','coef_high']]
    # res.to_csv(save_file,sep='\t',index=False)
    # print(f"Finished saving results for offensive language change in {tweet_type}:{identity}")
    return

def run_api_offensive_regression_worker(rq, agg, est, identity, tweet_type, model_type):
    assert rq=='offensive-api'
    assert agg in ['mean','max','count']
    assert est in ['abs','rel']
    assert tweet_type=='tweet'
    assert model_type in ['poisson','gee','mixedlm']
    
    time_unit = 'tweet'
    time_unit_col = 'week_diff'
    unit_col = 'tweet_origin'
    
    # save_dir=f'/shared/3/projects/bio-change/results/experiments/activity-change/poisson/{rq}/results-{tweet_type}-{time_unit}-{agg}-{est}'
    save_dir=f'/shared/3/projects/bio-change/results/experiments/activity-change/{model_type}/{rq}/results-{tweet_type}-{time_unit}-{agg}-{est}'
    cov_file=f'/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/mentions-from-api/sampled-covariates/all_covariates.{identity}.df.tsv'
    tweet_dir='/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/mentions-from-api/interactions-by-identity'
    identity_score_dir='/shared/3/projects/bio-change/data/interim/treated-control-propensity-tweets/mentions-from-api/identity-scores/'    
    # save_dir=f'/scratch/drom_root/drom0/minje/bio-change/06.regression/identity_added-with_tweet_identity/save_dir/{rq}/results-{tweet_type}-{time_unit}-{agg}-{est}'
    # cov_file=f'/scratch/drom_root/drom0/minje/bio-change/06.regression/identity_added-with_tweet_identity/cov_dir/all_covariates.{identity}.df.tsv'
    # tweet_dir='/scratch/drom_root/drom0/minje/bio-change/06.regression/identity_added-with_tweet_identity/tweet_dir'
    # identity_score_dir='/scratch/drom_root/drom0/minje/bio-change/06.regression/identity_added-with_tweet_identity/score_dir'
    # offensive_score_dir='/scratch/drom_root/drom0/minje/bio-change/06.regression/identity_added-with_tweet_identity/offensive_dir'
    
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    for file in os.listdir(save_dir):
        if file.startswith(f'table.{identity}'):
            print(f'File {file} already exists in {save_dir}! Skipping...')
            return

    # load covariates
    df_cov=pd.read_csv(cov_file,sep='\t',dtype={'user_id':str}).rename(columns={'is_identity':'in_treatment_group'})
    df_cov.head(2)
    
    # load all tweet info and match to score
    df_tweet=pd.read_csv(join(tweet_dir,f'{identity}.df.tsv'),sep='\t',dtype={'user_origin':str})
    df_tweet=df_tweet.drop(columns=['user_id'],axis=1).rename(columns={'user_origin':'user_id'})
    
    # add identity scores for text (replies by others) and text_origin (original tweet)
    for text_type in ['text_origin','text']:
        with open(join(identity_score_dir,f'{text_type}.{identity}.txt')) as f:
            scores=[float(x) for x in f.readlines()]
            df_tweet[f'identity_{text_type}_score']=scores
            df_tweet.loc[df_tweet[text_type].isnull(),f'identity_{text_type}_score']=0
            
    df_tweet=df_tweet[(df_tweet.is_english==True)&(df_tweet.is_english_origin==True)]

    # drop duplicates
    df_tweet=df_tweet.drop_duplicates()

    # get placeholder for each user and each week difference
    if time_unit_col=='month_diff':
        df_tweet=df_tweet[(df_tweet.month_diff>=-1)&(df_tweet.month_diff<=3)]
    if time_unit_col=='week_diff':
        df_tweet=df_tweet[(df_tweet.week_diff>=-4)&(df_tweet.week_diff<=12)]

    # merge (1) tweet counts, (2) identity tweet, (3) offensive tweet, (4) offensive tweet counts

    # valid columns to aggregate on
    valid_columns=['user_id',unit_col]
    # count of all origin tweets
    df1=df_tweet[['user_id','tweet_origin','week_diff','week_curr_origin','offensive_score_origin','identity_text_origin_score']].drop_duplicates()
    # count of all replies
    df2=df_tweet[valid_columns+['text']].groupby(valid_columns).count().reset_index()
    df2.columns=valid_columns+['count_replies']
    # count of identity-specific replies
    df3=df_tweet[df_tweet.identity_text_score>=0.5][valid_columns+['text']].groupby(valid_columns).count().reset_index()
    df3.columns=valid_columns+['count_identity_replies']
    # count of offensive replies
    df4=df_tweet[df_tweet.offensive_score>=0.5][valid_columns+['text']].groupby(valid_columns).count().reset_index()
    df4.columns=valid_columns+['count_offensive_replies']

    df5=df1.merge(df2,on=valid_columns,how='left').fillna(0)
    df5=df5.merge(df3,on=valid_columns,how='left').fillna(0)
    df5=df5.merge(df4,on=valid_columns,how='left').fillna(0)

    df6=df_cov.merge(df5,on=['user_id'],how='inner')

    # remove time unit corresponding to zero - needed because we want to remove the week of treatment which may be volatile and doesn't have enough data when aggregated at monthly basis

    # create additional column corresponding to (1) post-treatment time & (2) in treated group
    df6['treatment_effect']=(df6['in_treatment_group']*(df6['week_diff']>0)).astype(int)
    # create additional column corresponding to offensiveness or identity scores of treated users only
    df6['offensive_score_origin x treatment_effect']=df6['treatment_effect']*df6['offensive_score_origin']
    df6['identity_text_origin_score x treatment_effect']=df6['treatment_effect']*df6['identity_text_origin_score']    
    
    reg_columns = [
        'fri','fol','sta', # user activity history
        'profile_score', # identity score of previous profile
        'n_days_since_profile', # number of days since account was created
        'in_treatment_group', # is treated user
        'offensive_score_origin',
        'identity_text_origin_score',
        'count_replies',
        'offensive_score_origin x treatment_effect', # added effect coming from offensive scores of treated user only
        'identity_text_origin_score x treatment_effect', # added effect coming from identity scores of treated user only
        'treatment_effect', # has been treated
        ]

    # set up a regression task using statsmodels
    scaler = StandardScaler()
    X = pd.concat(
        [
            df6[reg_columns],
            pd.get_dummies(df6['week_treated'],prefix='week_treated',drop_first=True), # week for when profile was updated
            pd.get_dummies(df6[time_unit_col],prefix=time_unit_col,drop_first=True),
        ],
        axis=1
    )
    y = df6['count_offensive_replies']
    groups=df6['user_id']
    X[['fri','fol','sta']]=scaler.fit_transform(X[['fri','fol','sta']])
    X = sm.add_constant(X)

    # run model 
    if agg=='count':
        # model=sm.NegativeBinomial(endog=y, exog=X)
        # model=ZeroInflatedPoisson(endog=y, exog=X)
        # model=GeneralizedPoisson(endog=y, exog=X)
        # model = sm.GLM(endog=y, exog=X, family=sm.families.Poisson())
        # model = PoissonBayesMixedGLM(endog=y, exog=X, exog_vc=df3[['user_idxs']], ident=[0]) # doesn't work
        if model_type=='poisson':
            model=Poisson(endog=y, exog=X)
        elif model_type=='gee':
            model = GEE(endog=y, exog=X, family=sm.families.Poisson(), groups=groups)
        elif model_type=='mixedlm':
            model = sm.MixedLM(endog=y, exog=X, family=sm.families.Poisson(), groups=groups)
        result=model.fit()
        # gp_model = gpb.GPModel(group_data=df3['user_id'], likelihood="poisson")
        # gp_model.fit(y=y, X=X, params={'std_dev':True})
    else:
        model=sm.MixedLM(endog=y, exog=X, groups=groups)
        result=model.fit_regularized(maxiter=10000)
        
    print(result.summary2())
    save_file=join(save_dir,f'summary.{identity}.df.tsv')
    with open(save_file,'w') as f:
        f.write(str(result.summary2()))
        
    # save results as dataframe
    res=result.conf_int()
    res.columns=['coef_low','coef_high']
    res['coef']=result.params # coef
    res['pval']=result.pvalues
    res = res.reset_index()
    save_file=join(save_dir,f'table.{identity}.df.tsv')
    res = res[['index','pval','coef','coef_low','coef_high']]
    res.to_csv(save_file,sep='\t',index=False)
    print(f"Finished saving results for offensiveness change in {tweet_type}:{identity}")
        
    return

def run_regression(fun):
    inputs = []
    for tweet_type in ['tweet','retweet']:
        fun(tweet_type,mode='added')
    # fun('retweet','removed')
    return

def run_weekly_regression(idx=None):
    settings=[]
    inputs = []
    for est in ['abs']:
        # for time_unit in ['month','week']:
        for time_unit in ['month']:
            for rq in ['language']:
            # for rq in ['language','activity']:
                for model_type in ['gee']:
                # for model_type in ['gee','poisson']:
                    settings.append((est,time_unit,rq,model_type))
    if idx:
        settings=[settings[int(idx)]]

    identities = get_identities()
    
    print('settings:',settings)
    for est,time_unit,rq,model_type in settings:
        for agg in ['count']:
            for tweet_type in ['tweet','retweet']:
                for week in range(5,57):
                    inputs.append((rq, time_unit, week, agg, est, tweet_type, model_type))

# def run_weekly_regression_worker(rq, time_unit, week, agg, est, identity, tweet_type, model_type):

    pool = Pool(12)
    pool.starmap(run_weekly_regression_worker, inputs)

    # for input in inputs:
    #     run_weekly_regression_worker(*input)
    return

def run_past_regression(idx=None):
    identities = get_identities()
    settings=[]
    inputs = []
    for est in ['abs']:
        for time_unit in ['week']:
            for rq in ['language']:
                for model_type in ['gee']:
                # for model_type in ['gee','poisson']:
                    settings.append((est,time_unit,rq,model_type))
    if idx:
        settings=[settings[int(idx)]]

    print('settings:',settings)
    for est,time_unit,rq,model_type in settings:
        for agg in ['count']:
        # for agg in ['count','mean']:
                for tweet_type in ['tweet','retweet']:
                    for identity in identities:
                        inputs.append((rq, time_unit, agg, est, identity, tweet_type, model_type))
                    
    for input in inputs:
        run_regression_past_worker(*input)
    return

def run_offensive_regression(idx=None):
    identities = get_identities()
    settings = []
    rq='offensive'
    tweet_type='tweet'
    for est in ['abs']:
        for time_unit in ['week','month']:
            for model_type in ['gee']:
            # for model_type in ['gee','poisson']:
                settings.append((rq, time_unit, est, tweet_type, model_type))
                
    if idx:
        settings=[settings[int(idx)]]
    # rq, time_unit, agg, est, identity, tweet_type
    
    inputs = []
    for rq, time_unit, est, tweet_type, model_type in settings:
        for agg in ['count']:
            for identity in identities:
                inputs.append((rq, time_unit, agg, est, identity, tweet_type, model_type))
                
    for input in inputs:
        run_offensive_regression_worker(*input)
    return

def run_api_offensive_regression(idx=None):
    identities = get_identities()
    settings = []
    rq='offensive-api'
    tweet_type='tweet'
    for est in ['abs']:
        for time_unit in ['tweet']:
            # for model_type in ['gee']:
            for model_type in ['gee','poisson']:
                settings.append((rq, time_unit, est, tweet_type, model_type))
                
    if idx:
        settings=[settings[int(idx)]]
    # rq, time_unit, agg, est, identity, tweet_type
    
    inputs = []
    for rq, time_unit, est, tweet_type,model_type in settings:
        for agg in ['count']:
            for identity in identities:
                inputs.append((rq, agg, est, identity, tweet_type, model_type))
                
    for input in inputs:
        run_api_offensive_regression_worker(*input)
    return

if __name__=='__main__':
    # if len(sys.argv)>1:
    #     run_regression(sys.argv[1])
    # else:
    #     run_regression()
    # if len(sys.argv)>1:
    #     run_past_regression(sys.argv[1])
    # else:
    #     run_past_regression()
    # if len(sys.argv)>1:
    #     run_offensive_regression(sys.argv[1])
    # else:
    
    
    run_regression(fun=regression_total_count)
    run_regression(fun=regression_identity_count)
    run_regression(fun=regression_offensive_count)
    
    # run_past_regression()
    # run_api_offensive_regression()
    # run_offensive_regression()
    # run_weekly_regression()
