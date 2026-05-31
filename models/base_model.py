import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, save_path, seed, device):
        super().__init__()
        self.save_path = save_path
        self.seed = seed
        self.device = torch.device('cpu') if device < 0 else torch.device(f'cuda:{device}')
        self.model = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def get_criterion(self, *args, **kwargs):
        raise NotImplementedError

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    def load_model(self, *args, **kwargs):
        raise NotImplementedError