"""
Functions for collecting relevant Twitter data from decahose using the Greatlakes cluster
"""

import sys
from time import time,sleep
from multiprocessing import Pool
import re
import os
from os.path import join
import bz2
import gzip
from time import time

import ujson as json
import pandas as pd

def get_twitter_files():
    """ Return a list of twitter files from the decahose directory
    
    Returns:
        _type_: (list) String of twitter dates
    """
    twitter_file_names = []
    # period of our treated/control users: 2020-04 to 2021-04
    # base_dir='/nfs/turbo/twitter-decahose/decahose/raw/'
    base_dir='/scratch/drom_root/drom0/minje/bio-change/temp-tweets'
    candidates=sorted(os.listdir(base_dir))
    for file in candidates:
        if file.endswith('.bz2'):
            res=re.findall(r'decahose.([0-9]+)-([0-9]+)-(?:.*)\.bz2',file)
            if res:
                y,m=res[0]
                y,m=int(y),int(m)
                # if (y==2020) and (m in [4,5,6]):
                # if y in [2020,2021]:
                twitter_file_names.append(join(base_dir,file))
    print(f"{len(twitter_file_names)} files in total!")
    return twitter_file_names

def verify_tweet_of_interest(obj,uids):
    """Checks if the users affiliated with this tweet belong to our list of uids

    Args:
        obj (dict): Tweet object
        uids (set): Set containing all valid user ids

    Returns:
        _type_: (bool) Flag whether the tweet is associated with a valid user
    """
    # 1) if writer of tweet belongs to our list
    uid=obj['user']['id_str']
    if uid in uids:
        return True
    else:
        # if original user of the RT/quote tweet belongs to our list 
        for status in ['retweeted_status','quoted_status']:
            if status in obj:
                uid=obj[status]['user']['id_str']
                if uid in uids:
                    return True
        
        # if original user of the reply tweet belongs to our list
        uid=obj['in_reply_to_user_id_str']
        if uid:
            if uid in uids:
                return True
    return False

def get_user_info(obj):
    """Returns user info as a dictionary

    Args:
        obj (dict): Twitter user object

    Returns:
        _type_: (dict) Extracted user object
    """
    out={}
    for col in [
        'id_str','lang','name','screen_name','description',
               'verified','followers_count','friends_count','statuses_count','listed_count','favourites_count',
               'location','profile_image_url_https','profile_banner_url',
               'default_profile','default_profile_image',
               'profile_background_image_url_https',
               'created_at','utc_offset']:
        out[col]=obj[col]
    return out

def get_tweet_info(obj,out=None):
    """Returns extracted tweet info

    Args:
        obj (dict): Tweet object

    Returns:
        _type_: (dict) extracted tweet object
    """
    if out==None:
        out={}
    out['id']=obj['id_str']
    out['user_id']=obj['user']['id_str']
    out['lang']=obj['lang'] if 'lang' in obj else None
    out['text']=obj['extended_tweet']['full_text'] if 'extended_tweet' in obj else obj['text']
    out['created_at']=obj['created_at']
    out['tweet_type']='tweet'
    out['user_mentions']=[(x['screen_name'],x['name'],x['id_str']) for x in obj['entities']['user_mentions']]
    return out

def add_retweet_info(obj,out,status):
    """Returns extracted retweet info

    Args:
        obj (dict): Tweet object

    Returns:
        _type_: (dict) extracted tweet object
    """
    out['user_id_origin']=obj['user']['id_str']
    out['id_origin']=obj['id_str']
    out['created_at_origin']=obj['created_at']
    if status=='retweeted_status':
        out['lang']=obj['lang'] if 'lang' in obj else out['lang']
        out['text']=obj['extended_tweet']['full_text'] if 'extended_tweet' in obj else obj['text']
        out['tweet_type']='retweet'
    elif status=='quoted_status':
        out['lang_origin']=obj['lang'] if 'lang' in obj else None
        out['text_origin']=obj['extended_tweet']['full_text'] if 'extended_tweet' in obj else obj['text']
        out['tweet_type']='quote'
        out['user_mentions']=[(x['screen_name'],x['name'],x['id_str']) for x in obj['entities']['user_mentions']]
    return out

def add_reply_info(obj,out):
    """Returns extracted reply info

    Args:
        obj (dict): Tweet object

    Returns:
        _type_: (dict) extracted tweet object
    """
    out['user_id_origin']=obj['in_reply_to_user_id_str']
    out['tweet_type']='reply' if obj['in_reply_to_status_id_str'] else 'mention'
    return out

def collect_tweets(file, save_dir):
    """Loads 

    Args:
        file (_type_): _description_
    """
    tweets=[]
    start = time()

    # load the treated & control users
    # treated_dir='/scratch/drom_root/drom0/minje/bio-change/05.classifier-training-data/'
    # save_dir='/scratch/drom_root/drom0/minje/bio-change/05.classifier-training-data/all-tweets'

    start = time()
    print(f'Starting {file}')

    # load users
    valid_users=set()
    with open('/scratch/drom_root/drom0/minje/bio-change/01.treated-control-users/all_treated_users.tsv') as f:
        for line in f:
            line=line.split('\t')
            uid=line[1]
            valid_users.add(uid)
    with open('/scratch/drom_root/drom0/minje/bio-change/01.treated-control-users/all_potential_control_users.tsv') as f:
        for line in f:
            line=line.split('\t')
            uid=line[0]
            valid_users.add(uid)

    uid2user = {} # to store user info


    cnt = 0
    if file.endswith('.gz'):
        f = gzip.open(file,'rt')
    elif file.endswith('.bz2'):
        f = bz2.open(file,'rt')

    file = file.replace('tweets.','').strip()
    save_file=file.split('/')[-1].replace('.bz2','.gz')
    outf = gzip.open(join(save_dir, 'tweets.'+save_file), 'wt')

    try:
        for ln,line in enumerate(f):
            if ln>100000:
                break
            # if ln%1000000==0:
            #     print(f'{file.split("/")[-1]}\t{ln}\t{int(time()-start)} seconds!')
            # if ln>1000000:
            #     break

            # load object
            try:
                obj=json.loads(line)
            except:
                print("Error reading json! Skipping...")
                continue
            
            # test if object is affiliated with our users of interest
            try:
                flag = verify_tweet_of_interest(obj,valid_users)
            except:
                print("Error computing flag! Skipping...")
                continue
            if flag:
                cnt+=1
                uid = obj['user']['id_str']
                # get user info first
                if uid not in uid2user:
                    user = get_user_info(obj['user'])
                    uid2user[uid] = user
                # get tweet info
                try:
                    out = get_tweet_info(obj)
                except:
                    print("Error in get_tweet_info function!")
                    continue
                flag2 = True
                # get retweet info
                for status in ['retweeted_status', 'quoted_status']:
                    if status in obj:
                        uid = obj[status]['user']['id_str']
                        if uid not in uid2user:
                            user = get_user_info(obj[status]['user'])
                            uid2user[uid] = user
                        try:
                            out = add_retweet_info(obj[status], out, status)
                        except:
                            print("Error in add_retweet_info function!")
                            continue
                        flag2 = False
                        continue
                # else, plain tweet or reply
                if flag2:
                    # get reply info
                    if obj['in_reply_to_user_id_str']:
                        try:
                            out = add_reply_info(obj, out)
                        except:
                            print("Error in add_reply_info function!")
                            continue
                


                # get data at the end
                outf.write(json.dumps(out)+'\n')
                # tweets.append(out)
    except:
        pass

    f.close()
    outf.close()

    # save files
    with gzip.open(join(save_dir, 'users.'+save_file), 'wt') as outf:
        for obj in uid2user.values():
            outf.write(json.dumps(obj)+'\n')
    print(f'Completed {file.split("/")[-1]}\t{cnt}/{ln} lines!\t{int(time()-start)} seconds!')
    return

def collect_user_info(user_id_file, twitter_file, save_dir):
    """Collects user info object from the tweets generated between 2020.04.01-2020.05.01 

    Args:
        user_id_file (_type_): _description_
        twitter_file (_type_): _description_
        save_dir (_type_): _description_
    """
    tweets=[]
    users={}
    start = time()

    # load all user ids
    uids = set()
    with open(user_id_file) as f:
        for uid in f:
            uids.add(uid.strip())

    # dictionary that stores objects
    users = {}

    # read tweet file
    with bz2.open(twitter_file,'rt') as f:
        try:
            for ln,line in enumerate(f):
                try:
                    obj=json.loads(line)
                except:
                    print("Error reading json! Skipping...")
                    continue

                uid = obj['user']['id_str']

                # get user info first
                if uid in uids:
                    if uid not in users:
                        user = get_user_info(obj['user'])
                        users[uid] = user

                # get retweet info
                for status in ['retweeted_status', 'quoted_status']:
                    if status in obj:
                        uid = obj[status]['user']['id_str']
                        if uid in uids:
                            if uid not in users:
                                user = get_user_info(obj[status]['user'])
                                users[uid] = user
        except:
            pass

    # save files
    save_file=twitter_file.split('/')[-1].replace('.bz2','.gz')
    with gzip.open(join(save_dir, 'users.'+save_file), 'wt') as outf:
        for obj in users.values():
            outf.write(json.dumps(obj)+'\n')
    print(f'Completed {twitter_file.split("/")[-1]}\t{ln} lines!\t{int(time()-start)} seconds!')
    return

def test_fn(idx):
    sleep(5)
    # print(f'{idx} Slept 10 seconds!')
    return idx

def test_multiprocessing():
    import multiprocessing
    # Short example demonstrating how to determine the number of cores available.
    start=time()
    numCores = multiprocessing.cpu_count()
    print(f"I have {numCores} CPU cores available!")
    # example getting the number of available cpu cores
    from os import sched_getaffinity
    # get the number of available logical cpu cores
    n_available_cores = len(sched_getaffinity(0))
    # report the number of logical cpu cores
    print(f'Number of Available CPU cores: {n_available_cores}')
    pool=Pool(80)
    results=pool.map(test_fn,list(range(1600)))
    # print(sorted(results))
    print("Total time: ",int(time()-start))
    return

def set_multiprocessing(save_dir, files:list, modulo=None):
    """Script for running multiprocessing on greatlakes slurm.

    Args:
        modulo (_type_, optional): Module number to set (out of max 10). Defaults to None.
    """
    pool=Pool(32)
    from os import sched_getaffinity
    n_available_cores = len(sched_getaffinity(0))
    print(f'Number of Available CPU cores: {n_available_cores}')
    number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    print(f'Number of CPU cores: {number_of_cores}')

    if type(modulo)==str:
        modulo=int(modulo)
        files=[files[i] for i in range(len(files)) if i%10==modulo]
    print(len(files),' files to read!')

    inputs = []


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for twitter_file in files:
        inputs.append((twitter_file, save_dir))
    # try:
    # pool.map(collect_tweets,files)
    pool.starmap(collect_tweets,inputs)
    # finally:
    pool.close()
    return

def merge_user_files(start_dir, end_dir):
    # merge files
    uid2profile = {}
    for file in reversed(sorted(os.listdir(start_dir))):
        with gzip.open(join(start_dir,file),'rt')  as f:
            for line in f:
                obj=json.loads(line)
                uid = obj['id_str']
                if uid not in uid2profile:
                    uid2profile[uid] = obj
    
    # save file
    with gzip.open(join(end_dir,'user_profile-2020.04.json.gz'),'wt') as f:
        for uid,obj in uid2profile.items():
            f.write(json.dumps(obj)+'\n')
    return

if __name__=='__main__':
    # test_multiprocessing()
    # print("Start job")

    # files = get_twitter_files()
    file_dir = '/scratch/drom_root/drom0/minje/bio-change/01.treated-control-users/raw-tweets-2020'
    files = [join(file_dir,file) for file in sorted(os.listdir(file_dir))]
    save_dir='/scratch/drom_root/drom0/minje/bio-change/01.treated-control-users/all-tweets'
    if len(sys.argv)==2:
        set_multiprocessing(save_dir=save_dir, files=files, modulo=sys.argv[1])
    else:
        set_multiprocessing(save_dir=save_dir, files=files)

    # collect_tweets(
    #     file=files[0],
    #     user_id_file='/scratch/drom_root/drom0/minje/bio-change/05.classifier-training-data/labels_200000.df.tsv',
    #     save_dir='/scratch/drom_root/drom0/minje/bio-change/05.classifier-training-data/all-tweets'
    #     )
    
    # set_multiprocessing()
    # set_multiprocessing(sys.argv[1])

    # merge_user_files(
    #     start_dir='/scratch/drom_root/drom0/minje/bio-change/03.all-user-info/user_info',
    #     end_dir='/scratch/drom_root/drom0/minje/bio-change/03.all-user-info/')
