import numpy as np
import torch
import copy


class Repairer():
    def __init__(self, dataset, model_name, seed, device):
        self.dataset = dataset
        self.model_name = model_name
        self.seed = seed
        self.device = device
        self.error_types_train = dataset.error_types_train
    
    def repair(self):
        raise NotImplementedError


class TabularRepairer(Repairer):
    def __init__(self, dataset, model_name, seed, device):
        super().__init__(dataset, model_name, seed, device)
        self.task_type = dataset.task_type
        self.class_num = dataset.class_num

    def repair(self):
        dataset = copy.deepcopy(self.dataset)
        dataset = self._repair_label_errors(dataset)
        # dataset = self._repair_feature_errors(dataset)
        dataset = self._drop_feature_errors(dataset)
        sample_weights = self._repair_spurious_correlation()
        return dataset, sample_weights

    def _get_train_and_val_preds(self, k=3):
        from models.utils import get_model
        from sklearn.model_selection import StratifiedKFold

        X_train = self.dataset.X_train
        y_train = self.dataset.y_train
        X_val = self.dataset.X_val
        y_val = self.dataset.y_val
        output_dim = len(self.dataset.label_encoder)
        criterion = torch.nn.CrossEntropyLoss()
        y_dtype = torch.long
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.seed)
        train_preds_shape = (len(y_train), output_dim)
        val_preds_shape = (len(y_val), output_dim)
        oos_preds_train = np.zeros(train_preds_shape)
        val_preds_accum = np.zeros(val_preds_shape)
        epochs = 100
        split_arg = y_train
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, split_arg)):
            model = get_model(self.model_name, None, self.seed, self.device)
            model._build_model(X_train.shape[1], output_dim)
            model.to(self.device)
            X_train_fold = torch.tensor(X_train[train_idx], dtype=torch.float32).to(self.device)
            y_train_fold = torch.tensor(y_train[train_idx], dtype=y_dtype).to(self.device)
            X_val_fold_oos = torch.tensor(X_train[val_idx], dtype=torch.float32).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            model.train()
            for _ in range(epochs): 
                optimizer.zero_grad()
                outputs = model(X_train_fold)
                loss = criterion(outputs, y_train_fold)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                logits_oos = model(X_val_fold_oos)
                if self.task_type == 'cls':
                    preds_oos = torch.softmax(logits_oos, dim=1).cpu().numpy()
                    oos_preds_train[val_idx] = preds_oos
                else:
                    preds_oos = logits_oos.cpu().numpy().flatten()
                    oos_preds_train[val_idx] = preds_oos
                X_val_full = torch.tensor(X_val, dtype=torch.float32).to(self.device)
                logits_val = model(X_val_full)
                if self.task_type == 'cls':
                    preds_val = torch.softmax(logits_val, dim=1).cpu().numpy()
                else:
                    preds_val = logits_val.cpu().numpy().flatten()
                val_preds_accum += preds_val
        avg_val_preds = val_preds_accum / k            
        return oos_preds_train, avg_val_preds
    
    @staticmethod
    def _slice(X, mask):
        return X[mask]
    
    def _drop_label_errors(self, dataset):
        keep_mask = (self.error_types_train[:, 0] == 0)
        dataset.X_train, dataset.y_train, self.error_types_train = \
            self._slice(dataset.X_train, keep_mask), dataset.y_train[keep_mask], self.error_types_train[keep_mask]
        return dataset

    def _repair_label_errors(self, dataset):
        if self.task_type == 'reg' or self.class_num > 2:
            dataset = self._drop_label_errors(dataset)
        else:
            mask = (self.error_types_train[:, 0] == 1)
            y_repaired = dataset.y_train.copy()
            if self.class_num == 2:
                y_repaired[mask] = 1 - y_repaired[mask]
            else:
                train_preds, val_preds = self._get_train_and_val_preds()
                y_repaired[mask] =  np.argmax(train_preds[mask], axis=1)
            dataset.y_train = y_repaired
        return dataset

    def _drop_feature_errors(self, dataset):
        keep_mask = (self.error_types_train[:, 1] == 0)
        dataset.X_train, dataset.y_train, self.error_types_train = \
            self._slice(dataset.X_train, keep_mask), dataset.y_train[keep_mask], self.error_types_train[keep_mask]
        return dataset

    def _repair_feature_errors(self, dataset, threshold=5.0):
        error_types_train = self.error_types_train
        if error_types_train is None:
            return dataset
        clean_mask_train = (error_types_train[:, 1] == 0)
        X_clean_train = self._slice(dataset.X_train, clean_mask_train)
        y_clean_train = dataset.y_train[clean_mask_train]
        stats_map = {}
        if self.task_type == 'cls':
            unique_classes = np.unique(y_clean_train)
            for c in unique_classes:
                c_mask = (y_clean_train == c)
                if np.sum(c_mask) < 2:
                    stats_map[c] = (X_clean_train.mean(axis=0), X_clean_train.std(axis=0) + 1e-6)
                else:
                    X_c = X_clean_train[c_mask]
                    stats_map[c] = (X_c.mean(axis=0), X_c.std(axis=0) + 1e-6)
        else:
            stats_map['global'] = (X_clean_train.mean(axis=0), X_clean_train.std(axis=0) + 1e-6)
        
        def _apply_feature_repair(X, y, err_types):
            feat_err_mask = (err_types[:, 1] == 1)
            if not np.any(feat_err_mask):
                return X
            X_dirty = self._slice(X, feat_err_mask)
            y_dirty = y[feat_err_mask]
            target_means = np.zeros_like(X_dirty)
            target_stds = np.zeros_like(X_dirty)
            if self.task_type == 'cls':
                for c in np.unique(y_dirty):
                    if c in stats_map:
                        dirty_c_mask = (y_dirty == c)
                        target_means[dirty_c_mask] = stats_map[c][0]
                        target_stds[dirty_c_mask] = stats_map[c][1]
            else:
                target_means[:] = stats_map['global'][0]
                target_stds[:] = stats_map['global'][1]
            z_scores = np.abs((X_dirty - target_means) / target_stds)
            to_fix_mask = (z_scores > threshold)
            if np.sum(to_fix_mask) == 0:
                return X
            for col_idx in range(X.shape[1]):
                col_vals = X_clean_train[:, col_idx]
                is_integer = np.all(np.mod(col_vals, 1) == 0)
                row_indices = np.where(to_fix_mask[:, col_idx])[0]
                if len(row_indices) > 0:
                    vals = target_means[row_indices, col_idx]
                    if is_integer:
                        vals = np.round(vals)
                    X_dirty[row_indices, col_idx] = vals
            X_new = X.copy()
            X_new[feat_err_mask] = X_dirty
            return X_new
        dataset.X_train = _apply_feature_repair(dataset.X_train, dataset.y_train, self.error_types_train)
        return dataset
    
    def _repair_spurious_correlation(self):
        n_train = len(self.error_types_train)
        sample_weights = np.ones(n_train, dtype=np.float32)
        if self.error_types_train is not None:
            spurious_mask = (self.error_types_train[:, 2] == 1)
            sample_weights[~spurious_mask] = 0.5
        return sample_weights


class RecRepairer(TabularRepairer):
    def __init__(self, dataset, model_name, seed, device):
        super().__init__(dataset, model_name, seed, device)
    
    @staticmethod
    def _slice(X, mask):
        # X is a dict
        return {k: v[mask] for k, v in X.items()}