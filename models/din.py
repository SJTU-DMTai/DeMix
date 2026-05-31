import torch
from models.base_model import BaseModel
from models.deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint
from models.deepctr_torch.models.din import DIN as DeepCTR_DIN


# code based on deepctr_torch(https://github.com/shenweichen/DeepCTR-Torch)
class DIN(BaseModel):
    def __init__(self, save_path, seed, device):
        super().__init__(save_path, seed, device)
    
    def _build_model(self, dataset):
        feature_columns = [feat for feat in dataset.feature_columns if feat.name != 'error_type']
        self.model = DeepCTR_DIN(
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

    def forward(self, x):
        return self.model(x)

    def fit(self, dataset, batch_size=128, num_epochs=20, lr=0.0001, val=True, sample_weights=None):
        self._build_model(dataset)
        X_train, y_train = dataset.X_train, dataset.y_train
        X_val, y_val = dataset.X_val, dataset.y_val
        
        self.model.compile('adam', 'binary_crossentropy', lr=lr, metrics=['binary_crossentropy', 'auc'])

        if val:
            callbacks = [
                EarlyStopping(monitor='val_auc', patience=3, verbose=1, mode='max', min_delta=0.0001),
                ModelCheckpoint(filepath=self.save_path, monitor='val_auc', save_best_only=True, 
                                verbose=1, mode='max', save_weights_only=True)
            ]
            self.model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, 
                           epochs=num_epochs, verbose=1, callbacks=callbacks)
        else:
            self.model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=1)

    def evaluate(self, dataset):
        self.model.compile('adam', 'binary_crossentropy', metrics=['binary_crossentropy', 'auc'])
        eval_result = self.model.evaluate(dataset.X_test, dataset.y_test)
        print(f"Evaluation Result: {eval_result}")
        return eval_result

    def load_model(self, path=None):
        load_path = path if path is not None else self.save_path
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))
