
import os
import yaml
import pickle
from glob import glob
from os.path import abspath
from os.path import join as JP
from collections import defaultdict

def remove_zip_files(path):
    for fil in glob(JP(path,'*.zip')):
        os.remove(fil)

path = 'data/knowledge_dashboard/isocyanate_relevant'
path = 'data/knowledge_dashboard/isocyanate_irrelevant'
remove_zip_files(path)

def ensure_directories(input:str):
    ''' Create directory and subdirectories given a path as string '''
    if isinstance(input, dict):
        for k,path in input.items():
            if isinstance(path, dict):
                ensure_directories(path)
            else:
                ensure_by_level(path)
    elif isinstance(input, str):
        ensure_by_level(input)
    else:
        print('[ERROR]: Something weird in your paths')
    return


def ensure_by_level(levels, verbose=1):
    ''' Helper function for ensure_directories '''
    dir = ''
    for level in levels.split('/'):
        dir = JP(dir,level)
        if not os.path.exists(dir):
            if verbose == 1:
                print('Creating directory: ', dir)
            os.mkdir(dir)    
    return

    
def parse_yaml(path):
    with open(path, 'r') as f:
        params = yaml.safe_load(f)
        return params


def dict_to_yaml(data, dirpath, filename):
    err1 = 'Data must be type dict'
    err2 = 'Path does not exists: {}'.format(abspath(dirpath))
    assert type(data) == dict, err1
    assert os.path.exists(dirpath), err2
    with open(JP(dirpath,filename), 'w+') as f:
        yaml.dump(data, f)
    return


def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d