"""A file that contains the various scripts for merging different files
"""

import os
from os.path import join
import gzip
import re

import pandas as pd
import ujson as json
from tqdm import tqdm

# from twitter_identity.utils.utils import write_data_file_info

def merge_splitted_extracted_identities(load_dir,save_dir):
    """Merges the extracted identity shards

    Args:
        load_dir (_type_): _description_
        save_dir (_type_): _description_
    """
    
    for n_changes in ['0_changes','1plus_changes']:
        uid2profiles={}
        valid_files = [file for file in sorted(os.listdir(load_dir)) if n_changes in file]
        for file in tqdm(valid_files):
            identity = 'socialmedia' if 'social_media' in file else file.split('_')[-1]
            with open(join(load_dir,file),'r') as f:
                for line in f:
                    line=line.split('\t')
                    uid,ts,desc=line[:3]
                    desc=desc.strip()
                    if uid not in uid2profiles:
                        uid2profiles[uid] = {}
                    if ts not in uid2profiles[uid]:
                        uid2profiles[uid][ts]=[]
                    if desc:
                        uid2profiles[uid][ts].append(desc)
        
        # save
        print(f"Saving results for {n_changes}")
        with gzip.open(join(save_dir,f'description_changes.{n_changes}.all_identities.json.gz'),'wt') as outf:
            for uid,D1 in tqdm(uid2profiles.items()):
                times=sorted(D1.keys())
                for ts in times:
                    V = uid2profiles[uid][ts]
                    out_line = f'{uid}\t{ts}\t%s\n'%('\t'.join(V))
                    outf.write(out_line)
        # write_data_file_info(__file__, merge_splitted_extracted_identities.__name__, save_dir, [load_dir])
                    
    return

def merge_training_sets(user_id_file, load_dir, save_dir):
    """Loads all tweet files, and sorts them by each user, then by identity category
    """
    return

if __name__=='__main__':
    load_dir = '/shared/3/projects/bio-change/data/interim/description_changes/extracted/splitted'
    save_dir = '/shared/3/projects/bio-change/data/interim/description_changes/extracted/'
    merge_splitted_extracted_identities(load_dir, save_dir)
    