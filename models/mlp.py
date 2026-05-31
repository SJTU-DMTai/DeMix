import copy
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from models.base_model import BaseModel
from data.tabular_data import TabularDataset


class TabMLP(BaseModel):
    def __init__(self, save_path, seed, device):
        super().__init__(save_path, seed, device)

    def _build_model(self, input_dim, output_dim):
        pass
    
    def forward(self, X):
        return self.model(X)

    def get_criterion(self, dataset, reduction='mean'):
        task_type = dataset.task_type
        return nn.CrossEntropyLoss(reduction=reduction) if task_type == 'cls' else nn.MSELoss(reduction=reduction)

    def fit(self, dataset: TabularDataset, val=False, num_epochs=200, lr=1e-3, sample_weights=None):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed(self.seed)

        out_dim = len(dataset.label_encoder) if dataset.task_type == 'cls' else 1
        self._build_model(dataset.X_train.shape[1], out_dim)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        reduction = 'none' if sample_weights is not None else 'mean'
        criterion = self.get_criterion(dataset, reduction=reduction)
        X_train = torch.as_tensor(dataset.X_train, dtype=torch.float32, device=self.device)
        y_train = torch.as_tensor(dataset.y_train, device=self.device)
        w_train = torch.as_tensor(sample_weights, dtype=torch.float32, device=self.device) if sample_weights is not None else None
        if val:
            X_val = torch.as_tensor(dataset.X_val, dtype=torch.float32, device=self.device)
            y_val = torch.as_tensor(dataset.y_val, device=self.device)
        best_loss = float('inf')
        best_state = None

        for epoch in range(num_epochs):
            self.model.train()
            optimizer.zero_grad()
            preds = self.model(X_train)
            if dataset.task_type == 'reg':
                preds = preds.squeeze(1)
            loss = criterion(preds, y_train)
            if w_train is not None:
                loss = (loss * w_train).mean()
            loss.backward()
            optimizer.step()
            if val:
                self.model.eval()
                with torch.no_grad():
                    val_preds = self.model(X_val)
                    if dataset.task_type == 'reg':
                        val_preds = val_preds.squeeze(1)
                    val_loss = criterion(val_preds, y_val)
                    if len(val_loss.shape) >= 1:
                        val_loss = torch.mean(val_loss)

                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_state = copy.deepcopy(self.model.state_dict())
        if best_state:
            self.model.load_state_dict(best_state)
        
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save(self.model.state_dict(), self.save_path)

    def evaluate(self, dataset):
        if self.model is None: return
        self.model.eval()
        
        X_test = torch.as_tensor(dataset.X_test, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            preds = self.model(X_test)
            if dataset.task_type == 'reg':
                preds = preds.squeeze(1).cpu().numpy()
                mse = np.mean((preds - dataset.y_test) ** 2)
                print(f'Test MSE: {mse:.4f}')
            else:
                preds = torch.max(preds, 1)[1].cpu().numpy()
                acc = accuracy_score(dataset.y_test, preds)
                print(f'Test Accuracy: {acc:.4f}')

    def load_model(self, path, dataset):
        out_dim = len(dataset.label_encoder) if dataset.task_type == 'cls' else 1
        self._build_model(dataset.X_train.shape[1], out_dim)
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))


class TabMLP1(TabMLP):
    def __init__(self, save_path, seed, device):
        super().__init__(save_path, seed, device)

    def _build_model(self, input_dim, output_dim):
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        ).to(self.device)

        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)


class TabMLP2(TabMLP):
    def __init__(self, save_path, seed, device):
        super().__init__(save_path, seed, device)

    def _build_model(self, input_dim, output_dim):
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        ).to(self.device)


        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)