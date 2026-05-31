import os
import torch
from models.base_model import BaseModel
from data.base_data import BaseDataset
from demix.dec import DEClassifier
from demix.utils import get_influence, get_repairer


class Pipeline:
    def __init__(self, model_name: str, model: BaseModel, dataset: BaseDataset, seed: int, device: int):
        self.model_name = model_name
        self.model = model
        self.device = device
        self.seed = seed
        self.dataset = dataset
        self.save_dir = dataset.save_dir
        self.data_name = dataset.data_name
    
    def run(self):
        self.model.fit(self.dataset)
        self.model.evaluate(self.dataset)


class DeMixPipeline(Pipeline):
    def __init__(self, model_name: str, model: BaseModel, dataset: BaseDataset, seed: int, device: int, dec_path: str):
        super().__init__(model_name, model, dataset, seed, device)
        self.dec_path = dec_path
    
    def run(self):
        dec = DEClassifier(self.device, self.dec_path)
        dec.load_model()
        inf = get_influence(self.data_name)(self.model, self.dataset)
        inf_vec = inf.calc_influence_vectors()
        self.dataset.error_types_train = dec.predict(inf_vec)
        repairer = get_repairer(self.data_name)(self.dataset, self.model_name, self.seed, self.device)
        dataset, sample_weights = repairer.repair()
        self.model.fit(dataset, sample_weights=sample_weights)
        print('model performance after repair:')
        self.model.evaluate(dataset)
    
    def _get_model_names_for_dec(self, data_name):
        if data_name in ['adult', 'bank', 'credit', 'bike_sharing', 'air_quality', 'covertype']:
            model_names = ['mlp1', 'mlp2', 'ftt']
        else:
            raise ValueError(f'No model names configured for {data_name}')
        return model_names