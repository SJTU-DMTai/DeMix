import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import random
import os
from ucimlrepo import fetch_ucirepo
from data.base_data import BaseDataset


class TabularDataset(BaseDataset):
    def __init__(self, data_dir, data_name, seed):
        super().__init__(data_dir, data_name, seed)
        
        self.target_column = 'label'
        if data_name in ['adult', 'bank', 'credit', 'covertype']:
            self.task_type = 'cls'
        elif data_name in ['bike_sharing', 'air_quality']:
            self.task_type = 'reg'
        else:
            raise ValueError(f"Unsupported data_name: {data_name}")
        
        if not os.path.exists(os.path.join(self.data_dir, 'train.csv')):
            self._get_splited_data()
        self.load_raw_data()
        
        np.random.seed(self.seed)
        random.seed(self.seed)

    def _get_splited_data(self):
        os.makedirs(self.data_dir, exist_ok=True)
        id_map = {
            'adult': 2, 'bank': 222, 'credit': 350, 'covertype': 31,
            'bike_sharing': 275, 'air_quality': 360
        }
        dataset = fetch_ucirepo(id=id_map[self.data_name])
        X = dataset.data.features
        if self.data_name == 'air_quality':
            y = X[['C6H6(GT)']].copy()
            valid_mask = y['C6H6(GT)'] != -200
            X = X.loc[valid_mask].copy()
            y = y.loc[valid_mask].copy()
            X = X.drop(columns=['C6H6(GT)'])
        else:
            y = dataset.data.targets.copy()
        y.columns = [self.target_column]
        if self.data_name == 'adult':
            y[self.target_column] = y[self.target_column].astype(str).str.replace('.', '', regex=False).str.strip()
        if self.data_name == 'bike_sharing':
            X = X.drop(columns=['casual', 'registered'], errors='ignore')
        df = pd.concat([X, y], axis=1)
        split_args = {'test_size': 0.3, 'random_state': self.seed}
        if self.task_type == 'cls':
            y_enc = df[self.target_column].astype('category').cat.codes
            split_args['stratify'] = y_enc
        train_val_idx, test_idx = train_test_split(df.index, **split_args)
        if self.task_type == 'cls':
            y_tv = y_enc.loc[train_val_idx]
            split_args['stratify'] = y_tv if y_tv.nunique() > 1 else None
        split_args['test_size'] = 0.2
        train_idx, val_idx = train_test_split(train_val_idx, **split_args)
        for name, idx in zip(['train', 'valid', 'test'], [train_idx, val_idx, test_idx]):
            df.loc[idx].to_csv(os.path.join(self.data_dir, f'{name}.csv'), index=False)

    def load_raw_data(self):
        dfs = {mode: pd.read_csv(os.path.join(self.data_dir, f'{mode}.csv')) 
               for mode in ['train', 'valid', 'test']}
        
        self.feature_columns = [c for c in dfs['train'].columns if c != self.target_column]
        
        self._setup_preprocessor(dfs['train'][self.feature_columns])
        self._setup_target_processor(pd.concat([d[self.target_column] for d in dfs.values()]), 
                                     dfs['train'][self.target_column])
        
        self.X_train, self.y_train = self._process_data(dfs['train'])
        self.X_val, self.y_val = self._process_data(dfs['valid'])
        self.X_test, self.y_test = self._process_data(dfs['test'])
        self.class_num = len(self.label_encoder) if self.task_type == 'cls' else None
        print(f"train: {len(self.y_train)}, valid: {len(self.y_val)}, test: {len(self.y_test)}")

    def load_erroneous_data(self, df, mode='train'):
        if mode == 'train':
            et = df['error_type'].apply(lambda x: list(map(int, x)))
            self.error_types_train = np.array(et.tolist())
            df.drop(columns=['error_type'], inplace=True)
            self._setup_target_processor(df[self.target_column], df[self.target_column])
            self.X_train, self.y_train = self._process_data(df)
        elif mode == 'valid':
            et = df['error_type'].apply(lambda x: list(map(int, x)))
            self.error_types_val = np.array(et.tolist())
            df.drop(columns=['error_type'], inplace=True)
            self.X_val, self.y_val = self._process_data(df)

    def _process_data(self, df):
        X = df[self.feature_columns].copy()
        if 'datetime' in X.columns:
            X = self._extract_datetime_features(X)
        X_proc = self.preprocessor.transform(X)
        y_proc = self._process_target(df[self.target_column]).values
        return X_proc, y_proc

    def _setup_preprocessor(self, X):
        if 'datetime' in X.columns:
            X = self._extract_datetime_features(X)
        
        num_cols = X.select_dtypes(include=np.number).columns
        cat_cols = X.select_dtypes(include='object').columns
        
        self.preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ], remainder='passthrough').fit(X)

    def _extract_datetime_features(self, X):
        dt = pd.to_datetime(X['datetime'])
        X = X.assign(
            hour=dt.dt.hour, day_of_week=dt.dt.dayofweek,
            day_of_month=dt.dt.day, month=dt.dt.month,
            is_weekend=(dt.dt.dayofweek >= 5).astype(int)
        )
        return X

    def _setup_target_processor(self, all_y, train_y):
        self.label_encoder = None
        self.target_min, self.target_range = 0, 1
        
        if self.task_type == 'cls':
            unique_labels = sorted(all_y.unique())
            self.label_encoder = {l: i for i, l in enumerate(unique_labels)}
        else:
            self.target_min = float(train_y.min())
            self.target_max = float(train_y.max())
            self.target_range = (self.target_max - self.target_min) or 1.0

    def _process_target(self, y):
        if self.task_type == 'cls':
            return y.map(self.label_encoder)
        return (y.astype(np.float32) - self.target_min) / self.target_range

    def _denormalize_target(self, y_norm):
        if self.task_type == 'reg':
            return y_norm * self.target_range + self.target_min
        return y_norm

    def controlled_error_injection(self, mode='train', clean_ratio=0.5, separate=True):
        clean_ratio = round(clean_ratio, 1)
        out_dir = os.path.join(self.data_dir, 'perturb')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{mode}_s{self.seed}_r{clean_ratio}.csv')
        # if os.path.exists(out_path):
        #     df = pd.read_csv(out_path, dtype={'error_type': str})
        #     return df
        df = pd.read_csv(os.path.join(self.data_dir, f'{mode}.csv'))
        for col in ['is_label_error', 'is_feature_error', 'is_minority']:
            df[col] = 0
        df = self._construct_spurious_correlation(df)
        df = df.reset_index(drop=True)
        n_total = len(df)
        n_perturb_total = int(n_total * (1.0 - clean_ratio))
        n_label = n_perturb_total // 2
        n_feat = n_perturb_total - n_label
        all_indices = df.index.to_numpy()
        if separate:
            perm = np.random.permutation(all_indices)
            idx_flip = perm[: n_label]
            idx_feat = perm[n_label: n_label + n_feat]
        else:
            idx_flip = np.random.choice(all_indices, size=n_label, replace=False)
            idx_feat = np.random.choice(all_indices, size=n_feat, replace=False)
        df = self._apply_label_flip(df, idx_flip)
        df.loc[idx_flip, 'is_label_error'] = 1
        df = self._apply_feature_noise(df, idx_feat)
        df.loc[idx_feat, 'is_feature_error'] = 1
        df['error_type'] = df.apply(lambda row: ''.join(map(str, map(int, [row['is_label_error'], row['is_feature_error'], row['is_minority']]))), axis=1)
        df.drop(columns=['is_label_error', 'is_feature_error', 'is_minority'], inplace=True)
        df.to_csv(out_path, index=False)
        return df

    def _apply_label_flip(self, df, indices):
        if self.task_type == 'reg':
            y = df.loc[indices, self.target_column].values
            is_integer = np.all(y == y.astype(int))
            y_min = df[self.target_column].min()
            y_max = df[self.target_column].max()
            y_range = y_max - y_min
            perturbed_y = np.copy(y)
            for i in range(len(y)):
                val = y[i]
                min_diff = y_range * 0.25
                left_interval = (y_min, val - min_diff)
                right_interval = (val + min_diff, y_max)
                valid_intervals = []
                if left_interval[1] >= left_interval[0]:
                    valid_intervals.append(left_interval)
                if right_interval[1] >= right_interval[0]:
                    valid_intervals.append(right_interval)
                chosen_low, chosen_high = random.choice(valid_intervals)
                perturbed_val = random.uniform(chosen_low, chosen_high)
                perturbed_y[i] = perturbed_val
            if is_integer:
                perturbed_y = np.round(perturbed_y).astype(int)
            df.loc[indices, self.target_column] = np.clip(perturbed_y, 0, None)
            return df
        else:
            if len(indices) == 0:
                return df
            all_labels = df[self.target_column].unique()
            current_vals = df.loc[indices, self.target_column].to_numpy()
            rand_offsets = np.random.randint(1, len(all_labels), size=len(indices))
            label_to_idx = {lab: i for i, lab in enumerate(all_labels)}
            current_idx = np.array([label_to_idx[v] for v in current_vals])
            new_idx = (current_idx + rand_offsets) % len(all_labels)
            new_vals = all_labels[new_idx]
            df.loc[indices, self.target_column] = new_vals
            return df

    def _apply_feature_noise(self, df, indices):
        if len(indices) == 0:
            return df
        num_cols = df[self.feature_columns].select_dtypes(include=np.number).columns
        cat_cols = df[self.feature_columns].select_dtypes(exclude=np.number).columns
        feats = random.sample(self.feature_columns, min(random.randint(1, 5), len(self.feature_columns)))
        error_type = random.choice(['outlier', 'scale'])
        n = len(indices)
        idx_array = np.asarray(indices)
        for f in feats:
            if f in num_cols:
                col = df[f]
                if error_type == 'outlier':
                    mean_val, std_val = col.mean(), col.std()
                    signs = np.random.choice([-1, 1], size=n)
                    new_values = mean_val + signs * 5 * std_val
                    new_values = np.full(n, new_values) if np.isscalar(new_values) else new_values
                else:
                    scales = np.random.uniform(0.2, 5, size=n)
                    new_values = col.loc[idx_array].to_numpy() * scales
                if np.issubdtype(col.dtype, np.integer):
                    new_values = np.round(new_values).astype(col.dtype)
                df.loc[idx_array, f] = new_values
            elif f in cat_cols:
                col = df[f]
                uniques = col.unique()
                if len(uniques) <= 1:
                    continue
                current_vals = col.loc[idx_array].to_numpy()
                rand_offsets = np.random.randint(1, len(uniques), size=n)
                label_to_idx = {lab: i for i, lab in enumerate(uniques)}
                current_idx = np.array([label_to_idx[v] for v in current_vals])
                new_idx = (current_idx + rand_offsets) % len(uniques)
                df.loc[idx_array, f] = uniques[new_idx]
        return df

    def _construct_spurious_correlation(self, df, target_minority_ratio=0.15):
        all_rules = self._get_spurious_rules()
        selected_rule = all_rules[self.seed % len(all_rules)]
        conds = selected_rule['conditions']
        target = selected_rule['target']
        c_mask = pd.Series(True, index=df.index)
        for col, val in conds.items():
            if isinstance(val, tuple):
                c_mask &= (df[col] >= val[0]) & (df[col] <= val[1])
            elif isinstance(val, list):
                c_mask &= df[col].isin(val)
            else:
                c_mask &= (df[col] == val)
        if isinstance(target, tuple):
            t_mask_match = (df[self.target_column] >= target[0]) & (df[self.target_column] <= target[1])
        else:
            t_mask_match = (df[self.target_column] == target)
        mask_majority = (c_mask & t_mask_match) | ((~c_mask) & (~t_mask_match))
        mask_minority = ~mask_majority
        df['is_minority'] = 0
        df.loc[mask_minority, 'is_minority'] = 1
        idx_minority = df.index[mask_minority].to_numpy()
        idx_majority = df.index[mask_majority].to_numpy()
        current_minority = len(idx_minority)
        current_total = len(df)
        current_ratio = current_minority / current_total if current_total > 0 else 0
        # print(f"current ratio: {current_ratio}")
        if current_ratio < target_minority_ratio:
            if current_minority > 0 and target_minority_ratio > 0:
                n_drop = int(np.ceil(current_total - (current_minority / target_minority_ratio)))
                n_drop = min(n_drop, len(idx_majority)) 
                if n_drop > 0:
                    drop_indices = np.random.choice(idx_majority, size=n_drop, replace=False)
                    df = df.drop(index=drop_indices).reset_index(drop=True)
        elif current_ratio > target_minority_ratio:
            X = (current_minority - target_minority_ratio * current_total) / (1 - target_minority_ratio)
            n_drop = int(np.ceil(X))
            n_drop = min(n_drop, len(idx_minority))
            if n_drop > 0:
                drop_indices = np.random.choice(idx_minority, size=n_drop, replace=False)
                df = df.drop(index=drop_indices).reset_index(drop=True)
        return df

    def _get_spurious_rules(self):
        # rules for minority groups
        if self.data_name == 'adult':
            return [
                {'conditions': {'sex': 'Female', 'race': 'Black'}, 'target': '>50K'},
                {'conditions': {'race': ['Asian-Pac-Islander', 'Amer-Indian-Eskimo']}, 'target': '>50K'},
                {'conditions': {'occupation': 'Exec-managerial'}, 'target': '>50K'},
                {'conditions': {'workclass': 'Self-emp-inc'}, 'target': '>50K'},
            ]
        elif self.data_name == 'credit':
            return [
                {'conditions': {'X3': 1}, 'target': 1}, # Education
                {'conditions': {'X4': 1}, 'target': 1}, # Marital status
                {'conditions': {'X5': (50, 100)}, 'target': 1}  # Age
            ]
        elif self.data_name == 'bank':
            return [
                {'conditions': {'housing': 'no'}, 'target': 'yes'},
                {'conditions': {'marital': 'single'}, 'target': 'yes'},
                {'conditions': {'contact': 'cellular'}, 'target': 'yes'},
                {'conditions': {'poutcome': 'failure'}, 'target': 'yes'}
            ]
        elif self.data_name == 'covertype':
            return [
                {'conditions': {'Wilderness_Area4': 1}, 'target': 4}, 
                {'conditions': {'Elevation': (0, 2500)}, 'target': 4},
            ]
        elif self.data_name == 'bike_sharing':
            return [
                {'conditions': {'season': 1}, 'target': (600, 1000)}, 
                {'conditions': {'mnth': [12, 1, 2]}, 'target': (600, 1000)},
            ]
        elif self.data_name == 'air_quality':
            return [
                {'conditions': {'T': (30, 60)}, 'target': (15, 100)},
                {'conditions': {'RH': (0.8, 1)}, 'target': (15, 100)}
            ]
        else:
            raise ValueError(f"No spurious rules for {self.data_name}")