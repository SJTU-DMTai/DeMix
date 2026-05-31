from models.deepctr_torch.models.dien import DIEN as DeepCTR_DIEN
from models.din import DIN


class DIEN(DIN):
    def __init__(self, save_path, seed, device):
        super().__init__(save_path, seed, device)
    
    def _build_model(self, dataset):
        feature_columns = [feat for feat in dataset.feature_columns if feat.name != 'error_type']
        self.model = DeepCTR_DIEN(
            feature_columns, 
            dataset.behavior_feature_list,
            device=self.device, 
            att_weight_normalization=True,
            dnn_hidden_units=(64, 32),
            dnn_dropout=0.6,
            l2_reg_embedding=1e-3,
            l2_reg_dnn=1e-3,
            att_hidden_size=(64, 32),
            seed=self.seed
        )