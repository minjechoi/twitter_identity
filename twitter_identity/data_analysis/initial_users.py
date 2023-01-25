"""Mostly code that obtains initial numbers from the initial set of users
"""

import os
from os.path import join
from datetime import datetime
import gzip
from collections import Counter

from dateutil.parser import parse
import pandas as pd
import ujson as json
from tqdm import tqdm

def get_users_joined_in_range(user_file,start_date,end_date):
    """To get the number of users who have joined in a specific time period

    Args:
        user_file (str): Directory to the user dictionary file
        start_date (str): a date string
        end_date (str): a date string

    Returns:
        n_users (str): number of users fitting date range
    """
    
    dt_start = parse(start_date)
    dt_end = parse(end_date)
    
    n_users = 0
    with gzip.open(user_file,'rb') as f:
        for ln,_ in enumerate(f):
            continue
        
    with gzip.open(user_file,'rt') as f:
        pbar=tqdm(f,total=ln)
        for ln2,line in enumerate(pbar):
            # pbar.set_description(f'{n_users}/{ln2} within range')
            # if ln2>10000:
            #     break
            obj=json.loads(line)
            dt = parse(obj['created_at'],ignoretz=True)
            # print(dt)
            # print(dt_start)
            if (dt>=dt_start) and (dt<=dt_end):
                n_users+=1
    print(f'{n_users}/{ln} were created between {start_date} and {end_date}')
    return n_users
    
def get_language_of_users(user_file):
    """To get the language distribution of users

    Args:
        user_file (str): Directory to the user dictionary file

    Returns:
        cn (Counter): Counter object of language
    """
    
    user2lang = {}
    with gzip.open(user_file,'rb') as f:
        for ln,_ in enumerate(f):
            continue
        
    with gzip.open(user_file,'rt') as f:
        pbar=tqdm(f,total=ln)
        for ln2,line in enumerate(pbar):
            obj=json.loads(line)
            uid=obj['id_str']
            if uid not in user2lang:
                user2lang[uid]=obj['lang']
    
    cn = Counter(user2lang.values())
    print(cn.most_common())
    return cn
    
if __name__=='__main__':
    # get_users_joined_in_range(
    #     user_file='/shared/3/projects/bio-change/data/processed/user_profile-2020.04.json.gz',
    #     start_date='2020 April 1st', end_date='2020 April 30th')
    
    get_language_of_users(
    user_file='/shared/3/projects/bio-change/data/processed/user_profile-2020.04.json.gz'
    )
