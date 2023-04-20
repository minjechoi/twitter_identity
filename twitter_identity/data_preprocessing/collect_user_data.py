import os
from os.path import join
import gzip
import re
from random import sample
from collections import Counter
from multiprocessing import Pool
from datetime import datetime

from tqdm import tqdm
import pycld2 as cld2
import ujson as json
import pandas as pd
from dateutil.parser import parse

from twitter_identity.utils.utils import write_data_file_info, get_weekly_bins, strip_tweet, week_diff_to_month_diff

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
            for ids in identities:
                for identity in ids.split('|'):
                    id_ = identity.split(':')[0]
                    if id_ not in identity2uids:
                        identity2uids[id_]=set()
                    identity2uids[id_].add(uid)
    
    # for each identity, create positive and negative sets
    out=[]
    
    for id_ in sorted(identity2uids.keys()):
        S1=identity2uids[id_]
        S2 = all_uids-S1 # negative set of users
        print(id_,len(S1),len(S2))
        
        S1 = list(S1)
        S2 = list(S2)
        
        pos_users = S1 if len(S1)<max_users else sample(S1,max_users)
        neg_users = S2 if len(S2)<max_users else sample(S2,max_users)
        
        for uid in pos_users:
            out.append((uid,id_,1))
        for uid in neg_users:
            out.append((uid,id_,0))
        
    df = pd.DataFrame(out, columns=['user_id','identity','label'])
    df.to_csv(join(save_dir,f'labels_{max_users}.df.tsv'),sep='\t',index=False)
    write_data_file_info(__file__, identify_identity_positive_and_negative_users.__name__, save_dir, [user_file])
    
    # cn = Counter(D)
    # for k in sorted(cn.keys()):
    #     dn = round(cn[k]/len(D),3)
    #     print(f'{k}:{cn[k]}/{ln} ({dn})')
    return

def get_treated_users(description_file, extracted_file, save_dir):
    """Function for identifying treated users

    Args:
        load_file (_type_): _description_
        save_dir (_type_): _description_
    """
    
    # identify users who made only 1 change in their history
    uid2lines={}
    with gzip.open(description_file,'rt') as f:
        for ln,_ in enumerate(f):
            continue
            
    with gzip.open(description_file,'rt') as f:
        for line in tqdm(f,total=ln):
            uid,dt,desc=line.split('\t')
            if uid not in uid2lines:
                uid2lines[uid]=0
            uid2lines[uid]+=1
    S1=set([uid for uid,cnt in uid2lines.items() if cnt==2])
    print(f'{len(S1)} users who made only one change in the period!')
    
    # identify the identities of each user by each profile version
    with gzip.open(extracted_file,'rt') as f:
        for ln,_ in enumerate(f):
            continue

    uid2identities={}
    with gzip.open(extracted_file,'rt') as f:
        for line in tqdm(f,total=ln):
            line=line.strip().split('\t')
            uid,dt=line[:2]
            if uid in S1:
                if uid not in uid2identities:
                    uid2identities[uid]=[]
                identities=[]
                for id_ in line[2:]:
                    identities.extend(id_.split('|'))
                uid2identities[uid].append((float(dt),identities))

    valid_treated=[] # users who had zero identities posted but added something on their profiles
    valid_control=[] # users who had zero identities posted and did update their profiles but didn't add identity phrase
    for uid,(v1,v2) in tqdm(uid2identities.items()):
        dt1,arr1 = v1
        dt2,arr2 = v2
        if len(arr1)==0:
            if len(arr2)>0:
                valid_treated.append(uid)
            else:
                valid_control.append(uid)
    print(f'{len(valid_treated)} users who added an identity in their new profile!')
    
    # get users that can be correctly mapped to an identity     
    cat2treated = {}
    # test_uids = set(['1015817063938646016','1067904677285629953','1215409779583201280','4111265533'])
    for uid in tqdm(valid_treated):
        dt,arr = uid2identities[uid][1]
        arr = [x.split(':') for x in arr] # [[cat,phrases], [cat,phrases], ..., [cat,phrases]]
        if len(arr)==1:
            # one identity only - add straightaway
            cat=arr[0][0]
            if cat in ['political_anticonservative','political_antiliberal']:
                continue
            if cat not in cat2treated:
                cat2treated[cat]={}
            cat2treated[cat][uid]=(dt,arr[0][1])
        else:
            # multiple identity signals - look for conflicting identities
            # age - remove if multiple ages are included
            if len([cat for cat,phrases in arr if cat.startswith('age')])>1:
                arr = [x for x in arr if not x[0].startswith('age')]
                
            # gender - remove if men and women appear at the same time
            if len([cat for cat,phrases in arr if cat.startswith('gender')])>1:
                # consider as nonbinary
                phrases = ','.join([x[1] for x in arr if x[0].startswith('gender')])
                arr = [x for x in arr if not x[0].startswith('gender')] + [('gender_nonbinary',phrases)]
                
            # religion
            if len([cat for cat,phrases in arr if cat.startswith('religion')])>1:
                flag1 = len([cat for cat,phrases in arr if cat.startswith('religion')])==2
                flag2 = False
                for cat,_ in arr:
                    if cat=='religion_general':
                        flag2=True
                        break
                if not (flag1&flag2): # remove duplicate religion phrases
                    arr = [x for x in arr if not x[0].startswith('religion')]
                    
            # political - remove if user has antiliberal or anticonservative
            pol_cats = set([x[0] for x in arr if (x[0].startswith('political_') and (x[0]!='political_blm'))])
            flag1=False
            if ('political_anticonservative' in pol_cats) or ('political_antiliberal' in pol_cats):
                flag1=True
            flag2=False
            # political - remove if user has both conservative and liberal
            if ('political_conservative' in pol_cats) and ('political_liberal' in pol_cats):
                flag2=True
            # if uid in test_uids:
            #     test = [x[0] for x in arr]
            #     print(test)
            #     print(arr)
            #     print(pol_cats)
            #     print(flag1,flag2)
                
            if flag1 or flag2:
                arr = [x for x in arr if (x[0]=='political_blm') or (x[0].startswith('political')==False)]
                # arr = [x for x in arr if not ((x[0].startswith('political')) and (x[0]!='political_blm'))] # remove all political instances that are not blm
            
            # use remaining phrases (if any)
            for cat,phrases in arr:
                # if cat=='political_anticonservative':
                #     print(uid,arr)
                if cat not in cat2treated:
                    cat2treated[cat]={}
                cat2treated[cat][uid]=(dt,phrases)

    print("Saving treated users!")
    output_file = join(save_dir,'all_treated_users.tsv')
    cnt=0
    with open(output_file,'w') as f:
        for cat,D in tqdm(cat2treated.items()):
            for uid,(dt,phrases) in D.items():
                f.write(f'{cat}\t{uid}\t{dt}\t{phrases}\n')
                cnt+=1
    
    write_data_file_info(__file__,get_treated_users.__name__, output_file, [description_file,extracted_file])
    print(f"{cnt} treated users!")
    for cat in sorted(cat2treated.keys()):
        D=cat2treated[cat]
        print(f'{cat}:{len(D)} treated users')
    
    print("Saving potential control users!")
    output_file = join(save_dir,'all_potential_control_users.tsv')
    with open(output_file,'w') as f:
        for uid in tqdm(valid_control):
            dt,_ = uid2identities[uid][1] # dt at the time when the change was made
            f.write(f'{uid}\t{dt}\n')
    write_data_file_info(__file__,get_treated_users.__name__, output_file, [description_file,extracted_file])
        
    return    
    

def get_user_reply_data_for_classifier(user_identity_file, user_name_file, data_file, save_file):
    uid2week={}
    uid2data={}
    uid2identity={}
    # get list of all users to consider
    df=pd.read_csv(user_identity_file,sep='\t',dtype={'user_id':str})
    for identity,uid,ts in df.values:
        uid2week[uid]=get_weekly_bins(ts)
        uid2identity[uid]=identity

    # get files to map user name to id    
    username2uid={}
    with open(user_name_file) as f:
        for ln,line in enumerate(f):
            obj=json.loads(line)
            name=obj['username'].lower()
            uid=obj['id']
            if uid in uid2identity:
                username2uid[name]=uid
                
    out = []
    with open(data_file) as f:
        for line in f:
            flag=False
            obj=json.loads(line)
            text=obj['text']
            ts=obj['created_at'] 
            if 'in_reply_to_user_id' in obj:
                uid=obj['in_reply_to_user_id']
                if uid in uid2identity:
                    flag=True
            else:
                names = re.findall(r'@(\w+)',text.lower())
                for name in names:
                    if name in username2uid:
                        uid = username2uid[name]
                        flag=True
                        break
            if flag:
                text=strip_tweet(text)
                identity=uid2identity[uid]
                week_tweet = get_weekly_bins(ts)
                wd=week_tweet-uid2week[uid]
                md = week_diff_to_month_diff(wd)
                out.append((uid,identity,wd,md,text))
    df2=pd.DataFrame(out,columns=['user_id','identity','week_difference','month_difference','text'])
    df2.to_csv(save_file,sep='\t',index=False)
    write_data_file_info(__file__, get_user_reply_data_for_classifier.__name__, save_file, [data_file,user_identity_file,user_name_file])
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
    # identify_identity_positive_and_negative_users(
    #     user_file='/shared/3/projects/bio-change/data/interim/description_changes/extracted/description_changes.0_changes.all_identities.json.gz',
    #     save_dir='/shared/3/projects/bio-change/data/interim/identity_classifier-train_data/positive-negative-users/',
    #     max_users=200000
    # )
    
    # get_treated_users(
    #     description_file='/shared/3/projects/bio-change/data/interim/description_changes/filtered/description_changes_1plus_changes.tsv.gz', 
    #     extracted_file='/shared/3/projects/bio-change/data/interim/description_changes/extracted/description_changes.1plus_changes.all_identities.json.gz', 
    #     save_dir='/shared/3/projects/bio-change/data/interim/treated-control-users'
    # )
    
    user_identity_file = '/shared/3/projects/bio-change/data/interim/1000-samples-per-identity/sampled_uids.df.tsv'
    user_name_file = '/shared/3/projects/bio-change/data/interim/1000-samples-per-identity/users.json'
    data_file = '/shared/3/projects/bio-change/data/interim/1000-samples-per-identity/data.json'
    save_file = '/shared/3/projects/bio-change/data/interim/1000-samples-per-identity/replies.df.tsv'

    get_user_reply_data_for_classifier(user_identity_file, user_name_file, data_file, save_file)