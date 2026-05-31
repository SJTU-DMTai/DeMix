import copy
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from models.base_model import BaseModel

class NumericalFeatureTokenizer(nn.Module):
    def __init__(self, num_features, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_features, d_model))
        self.bias = nn.Parameter(torch.randn(num_features, d_model))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return x.unsqueeze(-1) * self.weight + self.bias

class FTTransformerModule(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, num_heads, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.tokenizer = NumericalFeatureTokenizer(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, 
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)
        
        nn.init.kaiming_uniform_(self.head.weight, a=math.sqrt(5))
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x_emb = self.tokenizer(x)
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_seq = torch.cat((cls_tokens, x_emb), dim=1)
        x_out = self.transformer(x_seq)
        cls_out = x_out[:, 0, :]
        cls_out = self.norm(cls_out)
        output = self.head(cls_out)
        return output

class FTTransformer(BaseModel):
    def __init__(self, save_path, seed, device):
        super().__init__(save_path, seed, device)
        self.model = None

    def forward(self, X):
        return self.model(X)

    def get_criterion(self, dataset, reduction='mean'):
        task_type = dataset.task_type
        return nn.CrossEntropyLoss(reduction=reduction) if task_type == 'cls' else nn.MSELoss(reduction=reduction)

    def fit(self, dataset, do_val=True, num_epochs=100, lr=1e-4, batch_size=1024, weight_decay=1e-5):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed(self.seed)

        output_dim = len(dataset.label_encoder) if dataset.task_type == 'cls' else 1
        input_dim = dataset.X_train.shape[1]

        self.model = FTTransformerModule(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=64,
            num_heads=2,
            num_layers=2,
            dim_feedforward=64,
            dropout=0.1
        ).to(self.device)

        label_dtype = torch.long if dataset.task_type == 'cls' else torch.float32
        criterion = self.get_criterion(dataset)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        X_train_tensor = torch.as_tensor(dataset.X_train, dtype=torch.float32)
        y_train_tensor = torch.as_tensor(dataset.y_train, dtype=label_dtype)
        train_ds = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        val_loader = None
        if do_val:
            X_val_tensor = torch.as_tensor(dataset.X_val, dtype=torch.float32)
            y_val_tensor = torch.as_tensor(dataset.y_val, dtype=label_dtype)
            val_ds = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        best_val_loss = float('inf')
        best_state = None
        
        for epoch in range(num_epochs):
            self.model.train()
            total_train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                preds = self.model(batch_X)
                if dataset.task_type == 'reg':
                    preds = preds.squeeze(1)
                
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item() * batch_X.size(0)

            avg_train_loss = total_train_loss / len(train_ds)
            
            current_val_loss = float('inf')
            if do_val and val_loader:
                self.model.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        preds = self.model(batch_X)
                        if dataset.task_type == 'reg':
                            preds = preds.squeeze(1)
                        loss = criterion(preds, batch_y)
                        total_val_loss += loss.item() * batch_X.size(0)
                
                current_val_loss = total_val_loss / len(val_ds)
                
                # if (epoch + 1) % 50 == 0 or epoch == 0:
                #     print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {current_val_loss:.4f}')

                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    best_state = copy.deepcopy(self.model.state_dict())

        if best_state:
            self.model.load_state_dict(best_state)
        
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save(self.model.state_dict(), self.save_path)

        # self.evaluate(dataset)

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
        output_dim = len(dataset.label_encoder) if dataset.task_type == 'cls' else 1
        input_dim = dataset.X_train.shape[1]
        
        self.model = FTTransformerModule(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=64,
            num_heads=2,
            num_layers=2,
            dim_feedforward=64,
            dropout=0.1
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))