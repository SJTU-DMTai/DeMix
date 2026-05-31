import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from demix.set_transformer import SetTransformerEncoder

PAD_VAL = -2

class DEClassifier(nn.Module):
    def __init__(self, device, save_path, input_channels=4, hidden_dim=128, feature_dim=128, num_heads=4, num_tasks=3, dropout=0.5):
        super().__init__()
        self.device = torch.device('cpu') if device < 0 else torch.device(f'cuda:{device}')
        self.save_path = save_path
        self.num_tasks = num_tasks
        self.pad_value = PAD_VAL
        self.best_ths = {i: 0.5 for i in range(num_tasks)}
        # Feature Extractor
        self.encoder = SetTransformerEncoder(
            input_dim=input_channels, hidden_dim=hidden_dim, output_dim=feature_dim,
            num_heads=num_heads, num_inds=32, num_seeds=1, ln=True, dropout=dropout
        )
        # Projection Head for Contrastive & Alignment Loss
        self.proj_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 32)
        )
        # Task Heads
        self.task_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(feature_dim, feature_dim // 2), nn.GELU(), nn.Linear(feature_dim // 2, 1))
            for _ in range(num_tasks)
        ])
        self.to(self.device)
    
    def load_model(self):
        checkpoint = torch.load(self.save_path, map_location=self.device, weights_only=True)
        self.load_state_dict(checkpoint['state_dict'])
        ths = checkpoint.get('best_ths', [0.5] * self.num_tasks)
        self.best_ths = {i: float(ths[i]) for i in range(len(ths))}

    def forward(self, x, return_proj=False):
        mask = (x[:, 0, :] != self.pad_value)
        feat = self.encoder(x.transpose(1, 2), mask=mask) 
        logits = torch.cat([head(feat) for head in self.task_heads], dim=1)
        if return_proj:
            return logits, self.proj_head(feat)
        return logits

    def _augment(self, X, intensity=0.5):
        # Augmentation to simulate different validation sets
        B, C, L = X.shape
        rand_idx = torch.rand(B, L, device=self.device).argsort(dim=1)
        shuffled = torch.gather(X, 2, rand_idx.unsqueeze(1).expand(-1, C, -1))
        keep_len = torch.randint(int(L * (1 - intensity)), L, (B, 1), device=self.device)
        mask = torch.arange(L, device=self.device).unsqueeze(0) < keep_len
        return torch.where(mask.unsqueeze(1), shuffled, torch.tensor(self.pad_value, device=self.device))

    def _contrastive_loss(self, z1, z2, temp=0.1):
        # InfoNCE on projected features (validation set invariance)
        z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
        sim_matrix = torch.matmul(z1, z2.T) / temp
        labels = torch.arange(z1.size(0), device=self.device)
        return F.cross_entropy(sim_matrix, labels)

    def _alignment_loss(self, feats, labels, model_ids):
        # Align class-conditional means across different models (model invariance)
        loss = 0
        unique_models = torch.unique(model_ids)
        if len(unique_models) < 2: return torch.tensor(0.0, device=self.device)
        for i in range(self.num_tasks):
            err_mask = labels[:, i] == 1 
            if err_mask.sum() == 0: continue
            z_err = feats[err_mask]
            m_err = model_ids[err_mask]
            centroids = []
            for m in unique_models:
                mask_m = m_err == m
                if mask_m.sum() > 0:
                    centroids.append(z_err[mask_m].mean(0))
            if len(centroids) > 1:
                centroids = torch.stack(centroids)
                center = centroids.mean(0)
                loss += F.mse_loss(centroids, center.expand_as(centroids))
        return loss

    def _focal_loss(self, logits, targets, weights, gamma=1):
        probs = torch.sigmoid(logits)
        bce = -weights * targets * torch.log(probs + 1e-7) - (1 - targets) * torch.log(1 - probs + 1e-7)
        focal = (1 - (targets * probs + (1 - targets) * (1 - probs))) ** gamma
        return (focal * bce).mean()
    
    def fit(self, dec_data, epochs=100, batch_size=256, lr=1e-3, lambda1=0.1, lambda2=0.1, val_split=0.2):
        vectors, labels, model_ids = dec_data
        vectors = vectors.cpu()
        pos_counts = np.maximum(labels.sum(0), 1)
        pos_w = torch.as_tensor((len(labels) - pos_counts) / pos_counts, dtype=torch.float32).to(self.device)
        pos_w = pos_w / 2
        # pos_w = torch.ones(3).to(self.device)
        print(f'Class weights: {pos_w.cpu().numpy()}')
        X_tr, X_val, y_tr, y_val, m_tr, m_val = train_test_split(
            vectors, labels, model_ids, test_size=val_split, stratify=labels, random_state=42
        )
        train_ds = TensorDataset(X_tr, y_tr, m_tr)
        val_ds = TensorDataset(X_val, y_val)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size)
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        best_f1 = 0.0
        for epoch in range(epochs):
            epoch_start = time.time()
            self.train()
            total_train_loss = 0.0
            for x, y, m in train_dl:
                x, y, m = x.to(self.device), y.to(self.device), m.to(self.device)
                v_aug = self._augment(x)
                v_cat = torch.cat([x, v_aug], dim=0)
                logits_cat, z_cat = self(v_cat, return_proj=True)
                B = x.size(0)
                l_pred, z_anc = logits_cat[:B], z_cat[:B]
                z_aug = z_cat[B:]
                loss_pred = sum(self._focal_loss(l_pred[:, i], y[:, i], pos_w[i]) for i in range(self.num_tasks))
                loss_v = self._contrastive_loss(z_anc, z_aug)
                loss_a = self._alignment_loss(z_anc, y, m)
                loss = loss_pred + lambda1 * loss_v + lambda2 * loss_a
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_train_loss += loss.item()
            sched.step()
            avg_train_loss = total_train_loss / len(train_dl)
            curr_f1s, best_ths = self.evaluate(val_dl)
            curr_f1_avg = np.mean(curr_f1s)
            epoch_time = time.time() - epoch_start
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], {epoch_time:.1f}s, train loss: {avg_train_loss:.3f}, LE: {curr_f1s[0]:.2f}, FE: {curr_f1s[1]:.2f}, SC: {curr_f1s[2]:.2f}")
            if curr_f1_avg > best_f1:
                best_f1 = curr_f1_avg
                torch.save({
                    'state_dict': self.state_dict(),
                    'best_ths': torch.as_tensor(best_ths, dtype=torch.float32),
                }, self.save_path)
                self.best_ths = best_ths
        self.load_model()
    
    def evaluate(self, loader):
        self.eval()
        preds, targs = [], []
        with torch.no_grad():
            for x, y in loader:
                p = self(x.to(self.device))
                preds.append(p.cpu())
                targs.append(y)
        probs = torch.sigmoid(torch.cat(preds)).numpy()
        labels = torch.cat(targs).numpy()
        f1s = []
        best_ths = [0.5] * self.num_tasks
        for i in range(self.num_tasks):
            best_f1 = 0
            best_th = 0.5
            for th in np.arange(0.3, 0.8, 0.05):
                p = (probs[:, i] > th).astype(int)
                tp = ((p == 1) & (labels[:, i] == 1)).sum()
                denom = p.sum() + labels[:, i].sum()
                current_f1 = 2 * tp / (denom + 1e-8)
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_th = th
            f1s.append(best_f1)
            best_ths[i] = best_th
        return f1s, best_ths
    
    def predict(self, vectors, batch_size=256):
        self.eval()
        if isinstance(vectors, np.ndarray): vectors = torch.FloatTensor(vectors)
        if vectors.ndim == 2: vectors = vectors.unsqueeze(1)
        all_preds = []
        with torch.no_grad():
            for i in range(0, vectors.size(0), batch_size):
                batch_vecs = vectors[i:i+batch_size].to(self.device)
                probs = torch.sigmoid(self(batch_vecs))
                preds = torch.zeros_like(probs, dtype=torch.int32)
                for j in range(self.num_tasks):
                    threshold = self.best_ths[j]
                    preds[:, j] = (probs[:, j] > threshold).int()
                all_preds.append(preds.cpu())
        return torch.cat(all_preds, dim=0).numpy()

    @staticmethod
    def prepare_training_data(save_dir, data_names, model_names, device, n_train, n_valid, n_perturb=5):
        from data.utils import get_dataset
        from models.utils import get_model
        from demix.utils import get_influence
        
        seed = 42
        all_vecs, all_labels, all_model_ids = [], [], []
        for m_idx, model_name in enumerate(model_names):
            for data_name in data_names:
                for j in range(n_perturb):
                    clean_ratio = round(0.4 + 0.1 * j, 1)
                    print(f'----------{model_name}, {data_name}, alpha={clean_ratio}----------')
                    cur_seed = seed + m_idx * 100 + j
                    dataset = get_dataset(save_dir, data_name, cur_seed)
                    d_train = dataset.controlled_error_injection(mode='train', clean_ratio=clean_ratio)
                    dataset.load_erroneous_data(d_train, mode='train')
                    d_valid = dataset.controlled_error_injection(mode='valid', clean_ratio=clean_ratio)
                    dataset.load_erroneous_data(d_valid, mode='valid')
                    model_path = f'{save_dir}/{data_name}/perturb_{model_name}.pth'
                    model = get_model(model_name, model_path, cur_seed, device)
                    model.fit(dataset)
                    n_t = min(len(dataset.X_train), n_train)
                    n_v = min(len(dataset.X_val), n_valid)
                    train_indices = np.random.choice(len(dataset.X_train), n_t, replace=False)
                    valid_indices = np.random.choice(len(dataset.X_val), n_v, replace=False)
                    inf = get_influence(data_name)(model, dataset, train_indices, valid_indices)
                    inf_vec = inf.calc_influence_vectors()
                    curr_labels = torch.as_tensor(dataset.error_types_train[train_indices])
                    all_vecs.append(inf_vec)
                    all_labels.append(curr_labels)
                    all_model_ids.append(torch.full((n_t,), m_idx, dtype=torch.long))
        # Pad vectors to max validation length
        max_l = max(v.shape[2] for v in all_vecs)
        padded_vecs = [F.pad(v, (0, max_l - v.shape[2]), value=PAD_VAL) for v in all_vecs]
        vectors_all = torch.cat(padded_vecs, dim=0)
        labels_all = torch.cat(all_labels, dim=0)
        model_ids_all = torch.cat(all_model_ids, dim=0)

        return vectors_all, labels_all, model_ids_all
