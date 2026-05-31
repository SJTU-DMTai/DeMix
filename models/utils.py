from models.mlp import TabMLP1, TabMLP2
from models.ftt import FTTransformer
from models.din import DIN
from models.dien import DIEN

def get_model(model_name, save_path, seed, device):
    if model_name == 'mlp1':
        return TabMLP1(save_path, seed, device)
    elif model_name == 'mlp2':
        return TabMLP2(save_path, seed, device)
    elif model_name == 'ftt':
        return FTTransformer(save_path, seed, device)
    elif model_name == 'din':
        return DIN(save_path, seed, device)
    elif model_name == 'dien':
        return DIEN(save_path, seed, device)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")