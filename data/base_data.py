import os

class BaseDataset:
    def __init__(self, save_dir, data_name, seed):
        self.save_dir = save_dir
        self.data_name = data_name
        self.seed = seed
        self.data_dir = os.path.join(save_dir, data_name)
    
    def load_raw_data(self):
        raise NotImplementedError
    
    def load_erroneous_data(self, mode='train'):
        raise NotImplementedError
    
    def controlled_error_injection(self, mode='train', clean_ratio=0.5):
        raise NotImplementedError