import os
import random
import numpy as np
import pandas as pd
import gzip
import zipfile
import shutil
import urllib.request
import json
from collections import defaultdict
from data.base_data import BaseDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from models.deepctr_torch.inputs import DenseFeat, SparseFeat, VarLenSparseFeat


class RecDataset(BaseDataset):
    def __init__(self, root_dir, data_name, seed,
                 min_seq_len: int = 1, max_seq_len: int = 100, neg_sample: int = 1):
        super().__init__(root_dir, data_name, seed)
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.neg_sample = neg_sample
        self.seed = seed
        self.task_type = 'cls'
        self.class_num = 2
        random.seed(seed)
        np.random.seed(seed)
        if not os.path.exists(os.path.join(self.data_dir, 'train.csv')):
            self._get_splited_data()
        self.load_raw_data()
    
    def _get_splited_data(self):
        user_interactions, user_features, item_features = self._download_data()
        user_interactions, user_features, item_features = self._encode_ids(
            user_interactions, user_features, item_features
        )
        self._split_by_time(user_interactions, user_features, item_features)

    def load_raw_data(self):
        dfs = {}
        for part in ['train', 'valid', 'test']:
            df = pd.read_csv(f'{self.data_dir}/{part}.csv')
            dfs[part] = df
        all_df = pd.concat(dfs.values(), ignore_index=True)
        self.id_cols = ['user_id', 'item_id']
        self.label_col = 'label'
        self.seq_col = 'seq'
        self.timestamp_col = 'timestamp'
        exclude_cols = set(self.id_cols + [self.label_col, self.seq_col, self.timestamp_col])
        self.cat_cols = []
        self.num_cols = []
        for col in all_df.columns:
            if col in exclude_cols:
                continue
            if all_df[col].dtype == 'object' or all_df[col].nunique() < 50:
                self.cat_cols.append(col)
            else:
                self.num_cols.append(col)
        self._encode_features(all_df)
        self._build_feature_columns(all_df)
        train_len, valid_len = len(dfs['train']), len(dfs['valid'])
        self.X_train, self.y_train = self._preprocess_deepctr(all_df.iloc[:train_len])
        self.X_val, self.y_val = self._preprocess_deepctr(all_df.iloc[train_len:train_len+valid_len])
        self.X_test, self.y_test = self._preprocess_deepctr(all_df.iloc[train_len+valid_len:])
        print("---- data summary ----")
        print(f"train: {len(self.y_train)}, valid: {len(self.y_val)}, test: {len(self.y_test)}")
        print("----------------------")
    
    def load_erroneous_data(self, df, mode='train'):
        if mode == 'train':
            self.X_train, self.y_train = self._preprocess_deepctr(df)
            pt = df['error_type'].apply(lambda x: list(map(int, x)))
            self.error_types_train = np.array(pt.tolist())
        elif mode == 'valid':
            self.X_val, self.y_val = self._preprocess_deepctr(df)
            pt = df['error_type'].apply(lambda x: list(map(int, x)))
            self.error_types_val = np.array(pt.tolist())
    
    def _split_by_time(self, user_interactions, user_features=None, item_features=None):
        all_samples = []
        for user, interactions in user_interactions.items():
            all_samples.extend(
                self._get_user_valid_samples(user, interactions, user_features, item_features)
            )
        all_samples.sort(key=lambda x: x['timestamp'])
        n = len(all_samples)
        train_end, val_end = int(n * 0.7), int(n * 0.85)
        train_raw = all_samples[:train_end]
        val_raw = all_samples[train_end:val_end]
        test_raw = all_samples[val_end:]
        train_data = self._keep_last_interaction(train_raw)
        val_data = self._keep_last_interaction(val_raw)
        test_data = self._keep_last_interaction(test_raw)
        self._process_and_save(train_data, val_data, test_data, user_interactions, 
                               user_features, item_features)
    
    def _process_and_save(self, train, val, test, interactions, u_feat, i_feat):
        train = self._add_negative_samples(train, interactions, u_feat, i_feat)
        val = self._add_negative_samples(val, interactions, u_feat, i_feat)
        test = self._add_negative_samples(test, interactions, u_feat, i_feat)
        pd.DataFrame(train).to_csv(f'{self.data_dir}/train.csv', index=False)
        pd.DataFrame(val).to_csv(f'{self.data_dir}/valid.csv', index=False)
        pd.DataFrame(test).to_csv(f'{self.data_dir}/test.csv', index=False)
        print(f"Split saved: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    
    def _encode_ids(self, user_interactions, user_features, item_features):
        all_users = set(user_interactions.keys())
        if user_features:
            all_users.update(user_features.keys())
        sample_user = next(iter(all_users), None)
        if isinstance(sample_user, str):
            print("Encoding User IDs...")
            user_map = {u: i for i, u in enumerate(sorted(all_users))}
            user_interactions = {user_map[u]: v for u, v in user_interactions.items() if u in user_map}
            if user_features:
                user_features = {user_map[u]: v for u, v in user_features.items() if u in user_map}
        all_items = set(getattr(self, 'all_items', set()))
        if item_features:
            all_items.update(item_features.keys())
        for seq in user_interactions.values():
            for _, item in seq:
                all_items.add(item)
        sample_item = next(iter(all_items), None)
        if isinstance(sample_item, str):
            print("Encoding Item IDs...")
            item_map = {item: i for i, item in enumerate(sorted(all_items))}
            for u in user_interactions:
                user_interactions[u] = [
                    (ts, item_map[item]) for ts, item in user_interactions[u] if item in item_map
                ]
            if item_features:
                item_features = {item_map[i]: v for i, v in item_features.items() if i in item_map}
            self.all_items = set(item_map.values())
        else:
            self.all_items = all_items
        return user_interactions, user_features, item_features

    def _get_user_valid_samples(self, user, interactions, user_features=None, item_features=None):
        samples = []
        u_feat = user_features.get(user, {}) if user_features else {}
        get_ifeat = item_features.get if item_features else lambda x, d={}: d
        history_items = []
        for timestamp, item in interactions:
            if len(history_items) >= self.min_seq_len:
                seq_items = history_items[-self.max_seq_len:]
                samples.append({
                    'user_id': user,
                    'item_id': item,
                    'timestamp': timestamp,
                    'seq': ','.join(map(str, seq_items)),
                    'label': 1,
                    **u_feat,
                    **get_ifeat(item, {})
                })
            history_items.append(item)
        return samples

    def _keep_last_interaction(self, data):
        return list({row['user_id']: row for row in data}.values())

    def _add_negative_samples(self, data_list, user_interactions, user_features=None, item_features=None):
        if self.neg_sample <= 0:
            return data_list
        all_items_list = list(self.all_items)
        n_all_items = len(all_items_list)
        user_interacted_sets = {u: {item for _, item in seq} for u, seq in user_interactions.items()}
        augmented_data = []
        get_ifeat = item_features.get if item_features else lambda x, d={}: d
        for row in data_list:
            augmented_data.append(row)
            interacted = user_interacted_sets.get(row['user_id'], set())
            neg_items = []
            if len(interacted) > n_all_items * 0.1:
                candidates = list(self.all_items - interacted)
                if candidates:
                    neg_items = random.sample(candidates, min(self.neg_sample, len(candidates)))
            else:
                while len(neg_items) < self.neg_sample:
                    t = random.choice(all_items_list)
                    if t not in interacted and t not in neg_items:
                        neg_items.append(t)
            for neg_item in neg_items:
                neg_row = row.copy()
                neg_row['item_id'] = neg_item
                neg_row['label'] = 0
                if item_features:
                    for k in get_ifeat(row['item_id'], {}):
                        neg_row.pop(k, None)
                    neg_row.update(get_ifeat(neg_item, {}))
                augmented_data.append(neg_row)
        return augmented_data

    def _encode_features(self, all_df):
        self.mappings = {}
        self.label_encoders = {}
        for col in self.id_cols:
            all_df[col] = all_df[col].astype(str)
            codes, uniques = pd.factorize(all_df[col])
            offset = 1 if col == 'item_id' else 0
            self.mappings[col] = dict(zip(uniques, np.arange(len(uniques)) + offset))
            le = LabelEncoder()
            le.classes_ = uniques
            self.label_encoders[col] = le
        for col in self.cat_cols:
            all_df[col] = all_df[col].fillna('__UNK__').astype(str).str.lower()
            codes, uniques = pd.factorize(all_df[col])
            self.mappings[col] = dict(zip(uniques, np.arange(len(uniques))))
            le = LabelEncoder()
            le.classes_ = uniques
            self.label_encoders[col] = le
        self.scaler = StandardScaler()
        if self.num_cols:
            all_df[self.num_cols] = all_df[self.num_cols].fillna(0)
            self.scaler.fit(all_df[self.num_cols])
        if self.timestamp_col in all_df.columns:
            self.ts_min = all_df[self.timestamp_col].min()
            self.ts_max = all_df[self.timestamp_col].max()

    def _build_feature_columns(self, all_df):
        embedding_dim = 8
        self.feature_columns = []
        for col in self.id_cols:
            vocab_size = len(self.mappings[col]) + (1 if col == 'item_id' else 0)
            self.feature_columns.append(SparseFeat(col, vocabulary_size=vocab_size, embedding_dim=embedding_dim))
        for col in self.cat_cols:
            self.feature_columns.append(SparseFeat(col, vocabulary_size=len(self.mappings[col]), embedding_dim=embedding_dim))
        for col in self.num_cols:
            self.feature_columns.append(DenseFeat(col, 1))
        if self.timestamp_col in all_df.columns:
            self.feature_columns.append(DenseFeat(self.timestamp_col, 1))
        self.feature_columns.append(
            VarLenSparseFeat(
                SparseFeat('hist_item_id', 
                           vocabulary_size=len(self.mappings['item_id']) + 1,
                           embedding_dim=embedding_dim,
                           embedding_name='item_id'), 
                maxlen=self.max_seq_len, 
                length_name="seq_length"
            )
        )
        self.behavior_feature_list = ["item_id"]

    def _preprocess_deepctr(self, df):
        model_input = {}
        for col in self.id_cols + self.cat_cols:
            model_input[col] = df[col].map(self.mappings[col]).fillna(0).astype(np.int32).values
        if self.num_cols:
            dense_vals = self.scaler.transform(df[self.num_cols].fillna(0))
            for i, col in enumerate(self.num_cols):
                model_input[col] = dense_vals[:, i]
        if self.timestamp_col in df.columns:
            denom = self.ts_max - self.ts_min + 1e-8
            model_input[self.timestamp_col] = ((df[self.timestamp_col] - self.ts_min) / denom).values
        seq_list, seq_length = self._process_sequences_deepctr(df[self.seq_col])
        model_input['hist_item_id'] = seq_list
        model_input['seq_length'] = seq_length
        return model_input, df[self.label_col].values

    def _process_sequences_deepctr(self, seq_series):
        item_mapping = self.mappings['item_id']
        seq_series = seq_series.fillna('')
        nested_seqs = seq_series.str.split(',').tolist()
        n_samples = len(nested_seqs)
        padded_seqs = np.zeros((n_samples, self.max_seq_len), dtype=np.int32)
        real_lengths = np.zeros(n_samples, dtype=np.int32)
        map_get = item_mapping.get
        max_len = self.max_seq_len
        for i, items in enumerate(nested_seqs):
            if not items or items == ['']:
                continue
            mapped = [x for x in (map_get(item, 0) for item in items) if x > 0]
            slen = len(mapped)
            if slen == 0:
                continue
            real_lengths[i] = min(slen, max_len)
            if slen <= max_len:
                padded_seqs[i, :slen] = mapped
            else:
                padded_seqs[i, :] = mapped[-max_len:]
        return padded_seqs, real_lengths

    def controlled_error_injection(self, mode='train', clean_ratio=0.5):
        out_dir = os.path.join(self.data_dir, 'perturb')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{mode}_s{self.seed}_r{clean_ratio}.csv')
        # if os.path.exists(out_path):
        #     df = pd.read_csv(out_path, dtype={'error_type': str})
        #     print("--- error type summary ---")
        #     print(df['error_type'].value_counts(normalize=True).sort_index())
        #     print("--------------------------")
        #     return df
        df = pd.read_csv(os.path.join(self.data_dir, f'{mode}.csv'))
        n_total = len(df)
        for col in ['is_label_error', 'is_feature_error', 'is_minority']:
            df[col] = 0
        df = self._construct_spurious_correlation(df)
        n_total = len(df)
        n_perturb_total = int(n_total * (1.0 - clean_ratio))
        n_bias = df['is_minority'].sum()
        n_remain = max(0, n_perturb_total - n_bias)
        n_label = n_remain // 2
        n_feat = n_remain - n_label
        candidates = df.index[df['is_minority'] == 0].to_numpy()
        if len(candidates) < (n_label + n_feat):
            candidates = df.index.to_numpy()
        perm = np.random.permutation(candidates)
        idx_flip = perm[:n_label]
        idx_feat = perm[n_label:n_label + n_feat]
        df = self._apply_label_flip(df, idx_flip)
        df.loc[idx_flip, 'is_label_error'] = 1
        df = self._apply_feature_noise(df, idx_feat)
        df.loc[idx_feat, 'is_feature_error'] = 1
        df['error_type'] = df.apply(lambda row: ''.join(map(str, map(int, [row['is_label_error'], row['is_feature_error'], row['is_minority']]))), axis=1)
        df.drop(columns=['is_label_error', 'is_feature_error', 'is_minority'], inplace=True)
        print("--- error type summary ---")
        print(df['error_type'].value_counts(normalize=True).sort_index())
        print("--------------------------")
        df.to_csv(out_path, index=False)
        return df

    def _apply_label_flip(self, df, indices):
        df.loc[indices, 'label'] = 1 - df.loc[indices, 'label']
        return df

    def _apply_feature_noise(self, df, indices):
        user_cols, item_cols = self._identify_user_item_features(df)

        if 'amazon' in self.data_name:
            target_cols = item_cols
        else:
            target_cols = random.sample(user_cols + item_cols, 1) if (user_cols + item_cols) else []
        
        if not target_cols and user_cols:
            target_cols = user_cols
        if not target_cols:
            return df
        
        entity_col = 'user_id' if target_cols == user_cols else 'item_id'
        affected_entities = df.loc[indices, entity_col].unique()
        n_feats = min(random.randint(1, 3), len(target_cols))
        feats_to_perturb = random.sample(target_cols, n_feats)
        
        stats = {}
        for feat in feats_to_perturb:
            if pd.api.types.is_numeric_dtype(df[feat]):
                stats[feat] = {
                    'type': 'numeric',
                    'mean': df[feat].mean(),
                    'std': df[feat].std(),
                    'is_int': pd.api.types.is_integer_dtype(df[feat])
                }
            else:
                stats[feat] = {'type': 'categorical', 'uniques': df[feat].unique()}
        self._perturb_features(df, feats_to_perturb, stats, entity_col, affected_entities)
        return df

    def _perturb_features(self, df, feats, stats, entity_col, entities):
        for feat in feats:
            s = stats[feat]
            if s['type'] == 'numeric':
                # entity_values = {e: (int(0) if s['is_int'] else 0.0) for e in entities}
                entity_values = {}
                for entity in entities:
                    val = s['mean'] + np.random.choice([-1, 1]) * 5 * s['std']
                    entity_values[entity] = int(round(val)) if s['is_int'] else val
                mask = df[entity_col].isin(entities)
                df.loc[mask, feat] = df.loc[mask, entity_col].map(entity_values)
            elif len(s['uniques']) > 1:
                entity_values = {}
                for entity in entities:
                    orig = df.loc[df[entity_col] == entity, feat].iloc[0] if len(df.loc[df[entity_col] == entity]) > 0 else None
                    if orig is not None:
                        choices = [x for x in s['uniques'] if x != orig]
                        if choices:
                            entity_values[entity] = random.choice(choices)
                if entity_values:
                    mask = df[entity_col].isin(entity_values.keys())
                    df.loc[mask, feat] = df.loc[mask, entity_col].map(entity_values)
    
    def _identify_user_item_features(self, df):
        exclude = set(self.id_cols + [self.label_col, self.seq_col, self.timestamp_col])
        feats = [c for c in df.columns if c not in exclude]
        return ([f for f in feats if f.startswith('user_')], 
                [f for f in feats if f.startswith('item_')])

    def _construct_spurious_correlation(self, df, target_minority_ratio=0.1):
        all_rules = self._get_spurious_rules()
        selected_rule = all_rules[self.seed % len(all_rules)]
        
        conds = selected_rule['conditions']
        target = selected_rule['target']
        
        c_mask = pd.Series(True, index=df.index)
        for col, val in conds.items():
            if col not in df.columns: continue
            if isinstance(val, tuple):
                c_mask &= (df[col] >= val[0]) & (df[col] <= val[1])
            elif isinstance(val, list):
                c_mask &= df[col].isin(val)
            else:
                c_mask &= (df[col] == val)
        
        if isinstance(target, tuple):
            t_mask_match = (df[self.label_col] >= target[0]) & (df[self.label_col] <= target[1])
        else:
            t_mask_match = (df[self.label_col] == target)
            
        mask_minority = ~((c_mask & t_mask_match) | ((~c_mask) & (~t_mask_match)))
        df['is_minority'] = 0
        df.loc[mask_minority, 'is_minority'] = 1
        
        idx_minority = df.index[mask_minority].to_numpy()
        idx_majority = df.index[~mask_minority].to_numpy()
        current_minority = len(idx_minority)
        current_total = len(df)
        current_ratio = current_minority / current_total if current_total > 0 else 0
        print(f'current ratio: {current_ratio}')
        if current_ratio < target_minority_ratio:
            n_drop = int(np.ceil(current_total - (current_minority / target_minority_ratio)))
            n_drop = min(n_drop, len(idx_majority))
            
            if n_drop > 0:
                drop_indices = np.random.choice(idx_majority, size=n_drop, replace=False)
                df = df.drop(index=drop_indices).reset_index(drop=True)

        elif current_ratio > target_minority_ratio:
            n_drop = int(np.ceil((current_minority - target_minority_ratio * current_total) / (1 - target_minority_ratio)))
            n_drop = min(n_drop, len(idx_minority))
            
            if n_drop > 0:
                drop_indices = np.random.choice(idx_minority, size=n_drop, replace=False)
                df = df.drop(index=drop_indices).reset_index(drop=True)
        return df

    def _get_spurious_rules(self):
        # rules for majority groups
        if self.data_name == 'amazon':
            return [
                    {'conditions': {'item_price': (0, 10)}, 'target': 1},
                    {'conditions': {'item_price': (500, 1000)}, 'target': 0},
                    {'conditions': {'item_brand': ['milbon', 'pre de provence', 'urban spa']}, 'target': 1},
                ]
        elif self.data_name == 'movielens':
            return [
                    {'conditions': {'user_gender': 'm', 'item_genres': ['action', 'sci-fi']}, 'target': 1},
                    {'conditions': {'user_gender': 'f', 'item_genres': ['romance', 'drama']}, 'target': 1},
                    {'conditions': {'user_age': (18, 30), 'item_genres': ['action', 'sci-fi']}, 'target': 1},
                ]
        elif self.data_name == 'yelp':
            return [
                    {'conditions': {'item_stars': (4.0, 5.1), 'item_review_count': (100, 10000)}, 'target': 1},
                    {'conditions': {'item_review_count': (0, 20), 'item_stars': (3.0, 4.0)}, 'target': 0},
                ]

    def _download_data(self):
        if self.data_name == 'amazon':
            return self._download_amazon()
        elif self.data_name == 'movielens':
            return self._download_movielens()
        elif self.data_name == 'yelp':
            return self._download_yelp()
        else:
            raise NotImplementedError(f'No download method implemented for {self.data_name}')
    
    def _download_amazon(self):
        # All_Beauty: https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/All_Beauty.json.gz
        # meta_All_Beauty: https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_All_Beauty.json.gz

        def download_and_extract(url, filename):
            target_path = os.path.join(self.data_dir, filename)
            if not os.path.exists(target_path):
                print(f"Downloading {filename} from {url}...")
                os.makedirs(self.data_dir, exist_ok=True)
                gz_path = target_path + '.gz'
                try:
                    urllib.request.urlretrieve(url, gz_path)
                    print(f"Extracting {filename}...")
                    with gzip.open(gz_path, 'rb') as f_in:
                        with open(target_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                except Exception as e:
                    print(f"Error downloading {filename}: {e}")
                    if os.path.exists(gz_path):
                        os.remove(gz_path)
                    if os.path.exists(target_path):
                        os.remove(target_path)
                    raise
                finally:
                    if os.path.exists(gz_path):
                        os.remove(gz_path)

        download_and_extract(
            'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/All_Beauty.json.gz', 'All_Beauty.json')
        
        download_and_extract(
            'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_All_Beauty.json.gz', 'meta_All_Beauty.json')

        meta_path = os.path.join(self.data_dir, 'meta_All_Beauty.json')
        print(f"Loading metadata from {meta_path}...")
        
        item_features = {}
        with open(meta_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                asin = item.get('asin')
                raw_price = item.get('price', '')
                item_price = None
                if raw_price:
                    try:
                        price_str = str(raw_price).replace('$', '').replace(',', '').strip()
                        item_price = float(price_str) if price_str else None
                    except:
                        item_price = None
                
                brand = item.get('brand', '')
                item_features[asin] = {
                    'item_price': item_price,
                    'item_brand': brand.lower() if isinstance(brand, str) else brand,
                }
                    
        # Load Reviews
        review_path = os.path.join(self.data_dir, 'All_Beauty.json')
        print(f"Loading reviews from {review_path}...")
        
        try:
            df = pd.read_json(review_path, lines=True)
            df = df[['reviewerID', 'asin', 'unixReviewTime', 'reviewerName']].rename(
                columns={'reviewerID': 'user_id', 'asin': 'item_id', 'unixReviewTime': 'timestamp'}
            )
        except ValueError:
            data = []
            with open(review_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        data.append([d['reviewerID'], d['asin'], d['unixReviewTime'], d.get('reviewerName', '')])
                    except:
                        pass
            df = pd.DataFrame(data, columns=['user_id', 'item_id', 'timestamp', 'reviewerName'])

        self.all_items = set(df['item_id'].unique())
        
        u_feat_df = df[['user_id', 'reviewerName']].drop_duplicates('user_id')
        u_feat_df['reviewerName'] = u_feat_df['reviewerName'].str.lower()
        user_features = u_feat_df.set_index('user_id')[['reviewerName']].rename(
            columns={'reviewerName': 'user_name'}
        ).to_dict('index')
        
        df = df.sort_values(['user_id', 'timestamp'])
        user_interactions = defaultdict(list)
        for uid, ts, iid in df[['user_id', 'timestamp', 'item_id']].values:
            user_interactions[uid].append((ts, iid))
            
        return user_interactions, user_features, item_features

    def _download_movielens(self):
        # https://files.grouplens.org/datasets/movielens/ml-1m.zip

        base_path = os.path.join(self.data_dir, 'ml-1m')
        if not os.path.exists(base_path):
            print("Downloading ml-1m dataset...")
            url = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
            zip_path = os.path.join(self.data_dir, 'ml-1m.zip')
            os.makedirs(self.data_dir, exist_ok=True)
            
            try:
                urllib.request.urlretrieve(url, zip_path)
                print("Extracting ml-1m.zip...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir)
            except Exception as e:
                print(f"Error downloading {zip_path}: {e}")
                if os.path.exists(base_path):
                    shutil.rmtree(base_path)
                raise
            finally:
                if os.path.exists(zip_path):
                    os.remove(zip_path)
        
        # Users
        users = pd.read_csv(os.path.join(base_path, 'users.dat'), sep='::', 
                            header=None, engine='python', encoding='latin-1',
                            names=['user_id', 'user_gender', 'user_age', 'user_occupation', 'user_zip_code'])
        users['user_gender'] = users['user_gender'].str.lower()
        users['user_zip_code'] = users['user_zip_code'].str.lower()
        user_features = users.set_index('user_id').to_dict('index')

        # Movies
        movies = pd.read_csv(os.path.join(base_path, 'movies.dat'), sep='::', 
                             header=None, engine='python', encoding='latin-1',
                             names=['item_id', 'item_title', 'item_genres'])
        movies = movies[['item_id', 'item_genres']]
        movies['item_genres'] = movies['item_genres'].str.lower()
        item_features = movies.set_index('item_id').to_dict('index')

        # Ratings
        ratings = pd.read_csv(os.path.join(base_path, 'ratings.dat'), sep='::', 
                              header=None, engine='python', encoding='latin-1',
                              names=['user_id', 'item_id', 'rating', 'timestamp'],
                              usecols=['user_id', 'item_id', 'timestamp'])
        
        self.all_items = set(ratings['item_id'].unique())
        ratings = ratings.sort_values(['user_id', 'timestamp'])
        
        user_interactions = defaultdict(list)
        for row in ratings.itertuples(index=False):
            user_interactions[row.user_id].append((row.timestamp, row.item_id))

        return user_interactions, user_features, item_features

    def _download_yelp(self):
        # import kagglehub
        # path = kagglehub.dataset_download("yelp-dataset/yelp-dataset")
        # print(path)
        
        # One should download the yelp dataset from https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset
        # and save it to the os.path.join(self.data_dir, "yelp_data") directory
        data_path = os.path.join(self.data_dir, "yelp_data")
        def load_json_df(fname, cols_map):
            path = os.path.join(data_path, fname)
            if not os.path.exists(path):
                return {}
            print(f"Loading {fname}...")
            df = pd.read_json(path, lines=True)
            return df.rename(columns=cols_map).set_index(list(cols_map.values())[0]).to_dict('index')

        # Users
        user_cols = {'user_id': 'user_id', 'average_stars': 'user_average_stars', 
                     'useful': 'user_useful', 'fans': 'user_fans'}
        user_features = load_json_df('yelp_academic_dataset_user.json', user_cols)

        # Items
        item_cols = {'business_id': 'item_id', 'city': 'item_city', 'state': 'item_state',
                     'stars': 'item_stars', 'review_count': 'item_review_count', 
                     'categories': 'item_categories'}
        item_features = load_json_df('yelp_academic_dataset_business.json', item_cols)
        
        # categories
        for item_id, features in item_features.items():
            for col in ['item_city', 'item_state']:
                if col in features and isinstance(features[col], str):
                    features[col] = features[col].lower()
                    
            categories_str = features.get('item_categories', '')
            cats = [c.strip().lower() for c in categories_str.split(',')] if isinstance(categories_str, str) else []
            
            for i in range(1, 6):
                features[f'item_cat{i}'] = cats[i-1] if i-1 < len(cats) else None
            features.pop('item_categories', None)

        # Reviews
        review_file = os.path.join(data_path, 'yelp_academic_dataset_review.json')
        print(f"Loading reviews from {review_file}...")
        
        df = pd.read_json(review_file, lines=True)[['user_id', 'business_id', 'date']]
        df['timestamp'] = pd.to_datetime(df['date']).astype(int) // 10**9
        
        self.all_items = set(df['business_id'].unique())
        df = df.sort_values(['user_id', 'timestamp'])
        
        user_interactions = defaultdict(list)
        for row in df[['user_id', 'timestamp', 'business_id']].itertuples(index=False):
            user_interactions[row[0]].append((row[1], row[2]))

        return user_interactions, user_features, item_features