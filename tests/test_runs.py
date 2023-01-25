import os
from os.path import join
import gzip
from datetime import datetime

from tqdm import tqdm

base_dir='/shared/2/projects/bio-change/data/4.description-changes/preprocessing'
S=set()

for file in sorted(os.listdir(base_dir)):
    if file.startswith('02.'):
        with gzip.open(join(base_dir,file),'rt') as f:
            for i,line in enumerate(f):
                # if i==0:
                #     continue
                uid=line.split()[0]
                S.add(uid)

    print(file,len(S))            