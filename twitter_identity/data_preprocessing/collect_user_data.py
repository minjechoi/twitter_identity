import os
from os.path import join
import gzip
from random import sample
from collections import Counter
from multiprocessing import Pool
from datetime import datetime

from tqdm import tqdm
import pycld2 as cld2
import ujson as json
import pandas as pd
from dateutil.parser import parse

from twitter_identity.utils.utils import write_data_file_info

def get_all_uids(input_files, output_file):
    
    S = set()
    for input_file in input_files:
        with gzip.open(input_file,'rt') as f:
            for i,line in enumerate(f):
                if i==0:
                    continue
                uid=line.split()[0]
                S.add(uid)

    with open(output_file,'w') as f:
        for line in S:
            f.write(line+'\n')
            
    write_data_file_info(__file__, get_all_uids.__name__, output_file, input_files)
    return

def remove_non_english_users(input_dir,output_dir):
    for file in sorted(os.listdir(input_dir)):
        
        with gzip.open(join(input_dir,file),'rt') as f:
            for ln,line in enumerate(f):
                continue
        
        non_eng_users = set()
        with gzip.open(join(input_dir,file),'rt') as f:
            for ln,line in enumerate(tqdm(f,total=ln)):
                if ln==0:
                    continue
                line=line.split('\t')
                uid,ts,desc=line[:3]
                try:
                    isReliable, textBytesFound, details = cld2.detect(desc)
                except:
                    continue
                for lang in details:
                    if lang[1] not in ['en','un']:
                        non_eng_users.add(uid)
                    
        with gzip.open(join(input_dir,file),'rt') as f,\
            gzip.open(join(output_dir,file),'wt') as outf:
            for ln,line in enumerate(tqdm(f,total=ln)):
                if ln==0:
                    continue
                uid=line.split('\t')[0]
                if uid not in non_eng_users:
                    outf.write(line)
        print(f'Finished saving {file}')
        write_data_file_info(__file__,remove_non_english_users.__name__,join(output_dir,file),[join(input_dir,file)])
    return
    
def remove_heavy_users(input_dir,output_dir,user_info_file):
    user2info = {}
    user_verified = set()
    
    print("Loading user info...")
    with gzip.open(user_info_file,'rt') as f:
        for ln,line in enumerate(f):
            continue
    with gzip.open(user_info_file,'rt') as f:
        for ln,line in enumerate(tqdm(f,total=ln)):
            obj=json.loads(line)
            uid=obj['id_str']
            user2info[uid]={
                'sta':obj['statuses_count'],
                'fol':obj['followers_count'],
                'fri':obj['friends_count'],
                }
            if obj['verified']:
                user_verified.add(uid)

    # identify users with most statuses & followers                
    all_statuses = sorted([(obj['sta'],uid) for uid,obj in user2info.items()])
    all_followers = sorted([(obj['fol'],uid) for uid,obj in user2info.items()])
    all_friends = sorted([(obj['fri'],uid) for uid,obj in user2info.items()])
    user_hi_sta = set([x[-1] for x in all_statuses[-int(ln/20):]])
    user_hi_fol = set([x[-1] for x in all_followers[-int(ln/20):]])
    user_hi_fri = set([x[-1] for x in all_friends[-int(ln/20):]])
    
    S = user_verified|user_hi_sta|user_hi_fol
    print(f"Identified {len(S)}/{ln} users of high influence!")
    
    # remove high status users
    for file in sorted(os.listdir(input_dir)):
        
        with gzip.open(join(input_dir,file),'rt') as f:
            for ln,line in enumerate(f):
                continue
                            
        with gzip.open(join(input_dir,file),'rt') as f,\
            gzip.open(join(output_dir,file),'wt') as outf:
            for ln,line in enumerate(tqdm(f,total=ln)):
                if ln==0:
                    continue
                uid=line.split('\t')[0]
                if uid not in S:
                    outf.write(line)
        print(f'Finished saving {file}')
        write_data_file_info(__file__,remove_heavy_users.__name__,join(output_dir,file),[join(input_dir,file),user_info_file])
    return

def filter_valid_users(input_dir,filter_dirs,output_dir):
    
    # get valid users obtained after filtering
    valid_users = set()
    for i,filter_dir in enumerate(filter_dirs):
        S=set()
        for file in sorted(os.listdir(filter_dir)):
            with gzip.open(join(filter_dir,file),'rt') as f:
                for ln,line in enumerate(f):
                    if ln==0:
                        continue
                    uid=line.split('\t')[0]
                    S.add(uid)
        
        if i==0:
            valid_users = S
        else:
            valid_users = valid_users & S
            
        print(f'Collected {len(valid_users)} users to preserve!')
    
    # save filtered users
    for file in sorted(os.listdir(input_dir)):
        
        with gzip.open(join(input_dir,file),'rt') as f:
            for ln,line in enumerate(f):
                continue
                            
        with gzip.open(join(input_dir,file),'rt') as f,\
            gzip.open(join(output_dir,file),'wt') as outf:
            for ln,line in enumerate(tqdm(f,total=ln)):
                if ln==0:
                    continue
                uid=line.split('\t')[0]
                if uid in valid_users:
                    outf.write(line)
        print(f'Finished saving {file}')
        write_data_file_info(__file__,filter_valid_users.__name__,join(output_dir,file),[join(input_dir,file)]+filter_dirs)
    return

def identify_identity_positive_and_negative_users(user_file,save_dir,max_users=100000):
    """Function that saves a list of identity positive- and negative- users with zero changes in their profiles - to be used for creating training datasets
    Args:
        load_dir (_type_): _description_
        save_dir (_type_): _description_
    """
    all_uids = set()
    identity2uids = {}
    with gzip.open(user_file,'rt') as f:
        for ln,line in enumerate(f):
            line=line.strip().split('\t')
            uid = line[0]
            identities = line[2:]
            all_uids.add(uid)
            for id in identities:
                id = id.split(':')[0]
                if id not in identity2uids:
                    identity2uids[id]=set()
                identity2uids[id].add(uid)
    
    # for each identity, create positive and negative sets
    out=[]
    
    for id in sorted(identity2uids.keys()):
        S1=identity2uids[id]
        S2 = all_uids-S1 # negative set of users
        print(id,len(S1),len(S2))
        
        S1 = list(S1)
        S2 = list(S2)
        
        pos_users = S1 if len(S1)<max_users else sample(S1,max_users)
        neg_users = S2 if len(S2)<max_users else sample(S2,max_users)
        
        for uid in pos_users:
            out.append((uid,id,1))
        for uid in neg_users:
            out.append((uid,id,0))
        
    df = pd.DataFrame(out, columns=['user_id','identity','label'])
    df.to_csv(join(save_dir,f'labels_{max_users}.df.tsv'),sep='\t',index=False)
    write_data_file_info(__file__, identify_identity_positive_and_negative_users.__name__, save_dir, [user_file])
    
    # cn = Counter(D)
    # for k in sorted(cn.keys()):
    #     dn = round(cn[k]/len(D),3)
    #     print(f'{k}:{cn[k]}/{ln} ({dn})')
    return


if __name__=='__main__':
    # get all uids from description files
    # get_all_uids(
    #     input_files=[
    #         '/shared/3/projects/bio-change/data/raw/description_changes_0_changes.tsv.gz',
    #         '/shared/3/projects/bio-change/data/raw/description_changes_1plus_changes.tsv.gz',
    #         ], 
    #     output_file='/shared/3/projects/bio-change/data/interim/all_uids/all_uids.txt')
    
    # remove non-english users
    # input_dir='/shared/3/projects/bio-change/data/raw/description_changes'
    # output_dir='/shared/3/projects/bio-change/data/interim/description_changes/remove-non-english'
    # remove_non_english_users(input_dir,output_dir)
    
    # remove heavy users
    # input_dir='/shared/3/projects/bio-change/data/raw/description_changes'
    # output_dir='/shared/3/projects/bio-change/data/interim/description_changes/remove-heavy-users'
    # user_info_file='/shared/3/projects/bio-change/data/raw/user_info/user_profile-2020.04.json.gz'
    # remove_heavy_users(input_dir,output_dir,user_info_file)
    
    # filter users
    # input_dir='/shared/3/projects/bio-change/data/raw/description_changes'
    # output_dir='/shared/3/projects/bio-change/data/interim/description_changes/filtered'
    # filter_dirs = [
    #     '/shared/3/projects/bio-change/data/interim/description_changes/remove-non-english',
    #     '/shared/3/projects/bio-change/data/interim/description_changes/remove-heavy-users'
    # ]
    # filter_valid_users(input_dir,filter_dirs,output_dir)

    # get positive and negative users
    identify_identity_positive_and_negative_users(
        user_file='/shared/3/projects/bio-change/data/interim/description_changes/extracted/description_changes.0_changes.all_identities.json.gz',
        save_dir='/shared/3/projects/bio-change/data/interim/identity_classifier-train_data/positive-negative-users/',
        max_users=200000
    )