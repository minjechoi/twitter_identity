import os
from os.path import join
import gzip

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
            
    write_data_file_info(__file__, get_all_uids.__name__, output_file, [input_file])
    return
    
if __name__=='__main__':
    get_all_uids(
        input_files=[
            '/shared/3/projects/bio-change/data/raw/description_0_changes.tsv.gz',
            '/shared/3/projects/bio-change/data/raw/description_1plus_changes.tsv.gz',
            ], 
        output_file='/shared/3/projects/bio-change/data/interim/user_data/all_uids.txt')