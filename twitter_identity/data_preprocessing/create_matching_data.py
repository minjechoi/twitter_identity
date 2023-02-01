import os
from os.path import join
import gzip

import pandas as pd
import ujson as json

from twitter_identity.utils.utils import get_weekly_bins

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
                    'name':obj['name'],
                    'screen_name':obj['screen_name']
                    }
    out = []
    for uid,V in uid2info.items():
        if V is not None:
            out.append((uid, V['fri'], V['fol'], V['sta'], V['created_at'], V['screen_name'], V['name']))
    df=pd.DataFrame(out,columns=['user_id','fri','fol','sta','created_at','screen_name','name'])
    save_file = join(save_dir,'user_activity_features.df.tsv')
    df.to_csv(save_file,sep='\t',index=False)
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
            if uid in uid2info:
                uid2info[uid].append((float(dt),desc))
    
    out = []
    for uid,V in uid2info.items():
        assert len(V)==2
        dt,desc = V[1][0], V[0][1]
        week = get_weekly_bins(dt)
        out.append((uid,dt,week,desc))
    df=pd.DataFrame(out,columns=['user_id','timestamp_treated','week_treated','profile_before_update'])
    save_file = join(save_dir,'description_features.df.tsv')
    df.to_csv(save_file,sep='\t',index=False)
    print(f'Saved to {save_file}')
    return

if __name__=='__main__':
    treat_user_file = '/shared/3/projects/bio-change/data/interim/treated-control-users/all_treated_users.tsv'
    control_user_file = '/shared/3/projects/bio-change/data/interim/treated-control-users/all_potential_control_users.tsv'
    save_dir = '/shared/3/projects/bio-change/data/interim/propensity-score-matching'
    user_info_file = '/shared/3/projects/bio-change/data/raw/user_info/user_profile-2020.04.json.gz'
    desc_info_file = '/shared/3/projects/bio-change/data/interim/description_changes/filtered/description_changes_1plus_changes.tsv.gz'

    valid_users = load_uids(treat_user_file, control_user_file)
    obtain_profile_features(valid_users, user_info_file, save_dir)
    obtain_description_features(valid_users, desc_info_file, save_dir)
    