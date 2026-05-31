import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.func import functional_call, vmap, grad


class Influence:
    def __init__(self, model, dataset, train_indices=None, valid_indices=None, batch_size=1024, max_valid_len=2048,):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_valid_len = max_valid_len
        self.train_indices = train_indices if train_indices is not None else list(range(len(self.dataset.X_train)))
        if valid_indices is None:
            valid_indices = list(range(len(self.dataset.X_val)))
        if len(valid_indices) > self.max_valid_len:
            valid_indices = valid_indices[:self.max_valid_len]
        self.valid_indices = valid_indices

    def calc_influence_vectors(self):
        raise NotImplementedError


class TabularInfluence(Influence):
    def __init__(self, model, dataset, train_indices=None, valid_indices=None, batch_size=1024, max_valid_len=2048):
        super().__init__(model, dataset, train_indices, valid_indices, batch_size, max_valid_len)
        self.device = model.device
        self.dataset = dataset
        self.sample_data()
    
    def sample_data(self):
        self.X_train_sub = torch.tensor(self.dataset.X_train[self.train_indices], dtype=torch.float32).to(self.device)
        self.X_val_sub = torch.tensor(self.dataset.X_val[self.valid_indices], dtype=torch.float32).to(self.device)
        self.y_train_sub = torch.tensor(self.dataset.y_train[self.train_indices], dtype=torch.float32).to(self.device)
        self.y_val_sub = torch.tensor(self.dataset.y_val[self.valid_indices], dtype=torch.float32).to(self.device)
        if self.dataset.task_type == 'cls':
            self.y_train_sub = self.y_train_sub.long()
            self.y_val_sub = self.y_val_sub.long()

    def calc_influence_vectors(self, use_aux=True):
        self.model.eval()
        valid_grads = []
        batch_size = self.batch_size
        for i in range(0, len(self.X_val_sub), batch_size):
            x_batch = self.X_val_sub[i:i+batch_size]
            y_batch = self.y_val_sub[i:i+batch_size]
            valid_grads.append(self.grad_z(x_batch, y_batch))
        all_valid_grads_T = torch.cat(valid_grads, dim=0).T 
        influence_vectors = []
        for i in range(0, len(self.X_train_sub), batch_size):
            x_batch = self.X_train_sub[i:i+batch_size]
            y_batch = self.y_train_sub[i:i+batch_size]
            train_grad_batch = self.grad_z(x_batch, y_batch)
            influence_batch = -1 * torch.matmul(train_grad_batch, all_valid_grads_T)
            influence_vectors.append(influence_batch)
        inf_vec = torch.cat(influence_vectors, dim=0)
        if use_aux:
            aux_vec = self.calc_aux_vectors()
            inf_vec = torch.stack([inf_vec] + aux_vec, dim=1)
        return inf_vec

    def calc_aux_vectors(self):
        X_train_norm = F.normalize(self.X_train_sub, p=2, dim=1)
        X_val_norm = F.normalize(self.X_val_sub, p=2, dim=1)
        sim_x = torch.matmul(X_train_norm, X_val_norm.T)
        if self.y_train_sub.dtype == torch.long:
            sim_y = (self.y_train_sub.view(-1, 1) == self.y_val_sub.view(1, -1)).float()
        else:
            sim_y = 1.0 / (1.0 + torch.abs(self.y_train_sub.view(-1, 1) - self.y_val_sub.view(1, -1)))
        
        self.model.eval()
        N = len(self.X_train_sub)
        M = len(self.X_val_sub)
        criterion = self.model.get_criterion(self.dataset, reduction='none')
        with torch.no_grad():
            train_logits = self.model(self.X_train_sub).view(N, -1)
            if self.y_train_sub.dtype == torch.long:
                train_losses = criterion(train_logits, self.y_train_sub)
            else:
                train_losses = criterion(train_logits.squeeze(), self.y_train_sub)
        sim_l = train_losses.view(-1, 1).expand(-1, M)

        return [sim_x, sim_y, sim_l]
        

    def grad_z(self, x, t, normalize=True):
        model = self.model
        criterion = model.get_criterion(self.dataset)
        params = {k: v for k, v in model.named_parameters() if v.requires_grad}
        def compute_loss(params, x_i, t_i):
            outputs = functional_call(model, params, x_i.unsqueeze(0))
            loss = criterion(outputs, t_i.unsqueeze(0))
            return loss
        batch_grad_fn = vmap(grad(compute_loss, argnums=0), in_dims=(None, 0, 0))
        per_sample_grads = batch_grad_fn(params, x, t)
        flat_grads = torch.cat([g.reshape(x.shape[0], -1) for g in per_sample_grads.values()], dim=1).detach()
        if normalize:
            return F.normalize(flat_grads, dim=1)
        return flat_grads


class RecInfluence(Influence):
    def __init__(self, model, dataset, train_indices=None, valid_indices=None, batch_size=256, max_valid_len=2048):
        super().__init__(model, dataset, train_indices, valid_indices, batch_size, max_valid_len)
        self.device = model.device
        self.dataset = dataset
        self.sample_data()

    def sample_data(self):
        feature_index = self.model.model.feature_index
        def _to_tensor(X, indices):
            x_list = []
            for feature in feature_index:
                arr = X[feature][indices]
                t = torch.tensor(arr, dtype=torch.float32).to(self.device)
                if t.dim() == 0:
                    t = t.unsqueeze(0).unsqueeze(1)
                elif t.dim() == 1:
                    t = t.unsqueeze(1)
                x_list.append(t)
            return torch.cat(x_list, dim=1)

        self.X_train_sub = _to_tensor(self.dataset.X_train, self.train_indices)
        self.X_val_sub = _to_tensor(self.dataset.X_val, self.valid_indices)
        self.y_train_sub = torch.tensor(self.dataset.y_train[self.train_indices], dtype=torch.float32).to(self.device)
        self.y_val_sub = torch.tensor(self.dataset.y_val[self.valid_indices], dtype=torch.float32).to(self.device)

    def calc_influence_vectors(self, use_aux=True):
        self.model.eval()
        valid_grads = []
        batch_size = self.batch_size
        for i in tqdm(range(0, len(self.X_val_sub), batch_size), desc="Calc validation grads"):
            x_batch = self.X_val_sub[i:i+batch_size]
            y_batch = self.y_val_sub[i:i+batch_size]
            valid_grads.append(self.grad_z(x_batch, y_batch))
        all_valid_grads_T = torch.cat(valid_grads, dim=0).T
        influence_vectors = []
        for i in tqdm(range(0, len(self.X_train_sub), batch_size), desc="Calc influence vectors"):
            x_batch = self.X_train_sub[i:i+batch_size]
            y_batch = self.y_train_sub[i:i+batch_size]
            train_grad_batch = self.grad_z(x_batch, y_batch)
            influence_batch = -1 * torch.matmul(train_grad_batch, all_valid_grads_T)
            influence_vectors.append(influence_batch.cpu())
        inf_vec = torch.cat(influence_vectors, dim=0)
        if use_aux:
            aux_vec = self.calc_aux_vectors()
            inf_vec = torch.cat([inf_vec] + aux_vec, dim=1)
        return inf_vec

    def calc_aux_vectors(self):
        X_train_norm = F.normalize(self.X_train_sub, p=2, dim=1)
        X_val_norm = F.normalize(self.X_val_sub, p=2, dim=1)
        sim_x = torch.matmul(X_train_norm, X_val_norm.T).cpu()
        sim_y = 1.0 / (1.0 + torch.abs(self.y_train_sub.view(-1, 1) - self.y_val_sub.view(1, -1)))
        sim_y = sim_y.cpu()

        self.model.eval()
        N = len(self.X_train_sub)
        M = len(self.X_val_sub)
        criterion = torch.nn.BCELoss(reduction='none')
        train_losses = []
        with torch.no_grad():
            for i in range(0, N, self.batch_size):
                x_batch = self.X_train_sub[i:i+self.batch_size]
                y_batch = self.y_train_sub[i:i+self.batch_size]
                outputs = self.model(x_batch).squeeze()
                losses = criterion(outputs, y_batch.squeeze())
                train_losses.append(losses)
        train_losses = torch.cat(train_losses, dim=0)
        loss_vec = train_losses.view(-1, 1).expand(-1, M).cpu()
        return [sim_x, sim_y, loss_vec]

    def grad_z(self, x, t, normalize=True):
        model = self.model
        criterion = torch.nn.BCELoss()
        params = {k: v for k, v in model.named_parameters()
                  if v.requires_grad and 'embedding' not in k.lower()}
        buffers = {k: v for k, v in model.named_buffers()}
        all_params = {k: v for k, v in model.named_parameters() if v.requires_grad}
        frozen_params = {k: v for k, v in all_params.items() if k not in params}

        def compute_loss(trainable_params, x_i, t_i):
            full_params = {**frozen_params, **trainable_params, **buffers}
            outputs = functional_call(model, full_params, x_i.unsqueeze(0))
            loss = criterion(outputs.squeeze(), t_i.squeeze())
            return loss

        batch_grad_fn = vmap(grad(compute_loss, argnums=0), in_dims=(None, 0, 0))
        per_sample_grads = batch_grad_fn(params, x, t)
        flat_grads = torch.cat([g.reshape(x.shape[0], -1) for g in per_sample_grads.values()], dim=1).detach()
        if normalize:
            return F.normalize(flat_grads, dim=1)
        return flat_grads
