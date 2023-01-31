import os
from datetime import datetime
import logging
import re
from typing import Literal

import numpy as np
from dateutil.parser import parse

def get_weekly_bins(timestamp):
    """
    A function that returns the number of the week based on starting date (2020.04.01)
    :param timestamp: the current timestamp
    :return:
    """
    dt_base = datetime(2020, 4, 1)
    try:
        dt_current = datetime.fromtimestamp(float(timestamp))
    except:
        dt_current = parse(timestamp)
        
    diff = dt_current - dt_base
    return diff.days / 7

_STRIP_TYPES = Literal[None, 'replace', 'remove']
def strip_tweet(text, url: _STRIP_TYPES='remove', username: _STRIP_TYPES='replace', hashtag: _STRIP_TYPES=None, non_ascii=True, lowercase=False, remove_rt=True):
    """Function that strips certain parts from text

    Args:
        text (str): Input string
        url (_STRIP_TYPES): _description_
        username (_STRIP_TYPES): _description_
    """
    # remove potential linebreaks and tabs
    text = ' '.join(text.replace('\n',' ').replace('\t',' ').split()).strip()
    
    # remove other languages
    if non_ascii:
        text = text.encode('ascii', errors='ignore').decode()
    
    # lowercase
    if lowercase:
        text = text.lower()
    
    # remove URLs
    if url=='replace':
        text = re.sub(r'http\S+', 'URL', text).strip()
    elif url=='remove':
        text = ' '.join(re.sub(r'http\S+', '', text).split()).strip()
    
    # remove usernames
    if username=='replace':
        text = re.sub(r'@\w+', '@username', text).strip()
    elif username=='remove':
        text = ' '.join(re.sub(r'@\w+', '', text).strip().split()).strip()
    
    # remove hashtags
    if hashtag=='replace':
        text = re.sub(r'#\w+', '#hashtag', text).strip()
    elif hashtag=='remove':
        text = ' '.join(re.sub(r'#\w+', '', text).strip().split()).strip()
    
    # remove retweet prefix (RT @username:)
    if remove_rt:
        text = re.sub(r'(?:RT|rt) @\w+:', '', text).strip()
        
    return text

def write_data_file_info(script_directory, function_name, output_file,  input_files=None, log_file='/home/minje/projects/twitter_identity/logs/log_files.tsv'
    ):
    """A function that writes the information used to create a file and logs it

    Args:
        script_directory (str): Directory of the script executed. Can be found using __file__
        function_name (str): Name of the function that is being run. Can be found using function_name.__name__
        output_file (str): Directory of the output file that is being stored
        input_files (list, optional): List of input file(s) if there are any.
        log_file (str, optional): Directory of log file. Defaults to '/home/minje/projects/twitter_identity/logs/log_files.tsv'.
    """
    if input_files:
        input_files=','.join(input_files)
    curr_time = str(datetime.now())
    with open(log_file,'a') as f:
        f.write('\t'.join([script_directory,function_name,output_file,input_files,curr_time])+'\n')
    return
    
def test_write_data_file_info():
    print(write_data_file_info(test_write_data_file_info))
    print(__file__)
    return

if __name__=='__main__':
    # get_weekly_bins()
    # print(test_write_data_file_info())
    text = "is it just me or does anyone else feel happy asf when you get a dm from someone saying \u201cyou\u201d or \u201cthis is you\u201d like damn bitch you thought of me\ud83e\udd7a\ud83e\udd7a\ud83e\udd7a\ud83e\udd7a\ud83e\udd7a"
    print(strip_tweet(text))
    