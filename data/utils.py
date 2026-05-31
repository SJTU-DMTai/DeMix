from data.tabular_data import TabularDataset
from data.rec_data import RecDataset

def get_dataset(root_dir, data_name, seed):
    if data_name in ['adult', 'bank', 'credit', 'students', 'diabetes', 'covertype', 'bike_sharing', 'air_quality']:
        return TabularDataset(root_dir, data_name, seed)
    elif data_name in ['amazon', 'movielens', 'yelp']:
        return RecDataset(root_dir, data_name, seed)
    else:
        raise ValueError(f"Unsupported data_name: {data_name}")