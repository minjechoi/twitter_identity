import os
from datetime import datetime
from dateutil.parser import parse
import logging

import numpy as np

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

def write_data_file_info(script_directory, function_name, output_file,  input_files=None, log_file='/home/minje/projects/tweet-identity/logs/log_files.tsv'
    ):
    """A function that writes the information used to create a file and logs it

    Args:
        script_directory (str): Directory of the script executed. Can be found using __file__
        function_name (str): Name of the function that is being run. Can be found using function_name.__name__
        output_file (str): Directory of the output file that is being stored
        input_files (list, optional): List of input file(s) if there are any.
        log_file (str, optional): Directory of log file. Defaults to '/home/minje/projects/tweet-identity/logs/log_files.tsv'.
    """
    if input_files:
        input_files=','.join(input_files)
    curr_time = datetime.now()
    with open(log_file,'a') as f:
        f.write('\t'.join([script_directory,function_name,output_file,input_files,curr_time])+'\n')
    return
    
def test_write_data_file_info():
    print(write_data_file_info(test_write_data_file_info))
    print(__file__)
    return

if __name__=='__main__':
    # get_weekly_bins()
    print(test_write_data_file_info())
    