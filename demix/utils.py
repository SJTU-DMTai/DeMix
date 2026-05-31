import random
import numpy as np
import torch
from demix.influence import TabularInfluence, RecInfluence
from demix.repair import TabularRepairer, RecRepairer

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_influence(data_name):
    if data_name in ['adult', 'bank', 'credit', 'covertype', 'bike_sharing', 'air_quality']:
        return TabularInfluence
    elif data_name in ['amazon', 'movielens', 'yelp']:
        return RecInfluence
    else:
        raise ValueError(f'No influence class implemented for {data_name}')

def get_repairer(data_name):
    if data_name in ['adult', 'bank', 'credit', 'covertype', 'bike_sharing', 'air_quality']:
        return TabularRepairer
    elif data_name in ['amazon', 'movielens', 'yelp']:
        return RecRepairer
    else:
        raise ValueError(f'No repairer class implemented for {data_name}')