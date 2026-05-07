class FeatureEngineer:

    def __init__(self):
        self.feature_names_created = []

    def parse_datetime_features(self, df):
        df = df.copy()
        for col in ['max_iso_eventtime_str', 'order_creation_dttm']:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception:
                    pass
        if 'order_creation_dttm' in df.columns and pd.api.types.is_datetime64_any_dtype(df['order_creation_dttm']):
            dt = df['order_creation_dttm']
            df['hour_of_day'] = dt.dt.hour
            df['day_of_week'] = dt.dt.dayofweek
            df['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
            df['is_night'] = ((dt.dt.hour >= 23) | (dt.dt.hour <= 5)).astype(int)
            df['is_business_hours'] = ((dt.dt.hour >= 9) & (dt.dt.hour <= 18)).astype(int)
            df['day_of_month'] = dt.dt.day
            df['is_month_start'] = (dt.dt.day <= 5).astype(int)
            df['is_month_end'] = (dt.dt.day >= 25).astype(int)
            self.feature_names_created.extend(['hour_of_day', 'day_of_week', 'is_weekend', 'is_night', 'is_business_hours', 'day_of_month', 'is_month_start', 'is_month_end'])
        return df

    def create_ratio_features(self, df):
        df = df.copy()
        if 'order_loan' in df.columns and 'loan_base_limit_part' in df.columns:
            df['loan_to_limit_ratio'] = df['order_loan'] / (df['loan_base_limit_part'] + 1e-08)
        debt_cols = [c for c in df.columns if c.startswith('debt_on_')]
        for col in debt_cols:
            ratio_name = f'{col}_to_loan_ratio'
            df[ratio_name] = df[col] / (df['order_loan'] + 1e-08)
            self.feature_names_created.append(ratio_name)
        if 'order_loan' in df.columns and 'sum_split_paid' in df.columns:
            df['current_to_historical_ratio'] = df['order_loan'] / (df['sum_split_paid'] + 1e-08)
            self.feature_names_created.append('current_to_historical_ratio')
        if 'score' in df.columns and 'mfm_score' in df.columns:
            df['score_x_mfm'] = df['score'] * df['mfm_score']
            df['score_minus_mfm'] = df['score'] - df['mfm_score']
            self.feature_names_created.extend(['score_x_mfm', 'score_minus_mfm'])
        if 'ml_limits_score_raw' in df.columns and 'score' in df.columns:
            df['ml_score_diff'] = df['ml_limits_score_raw'] - df['score']
            self.feature_names_created.append('ml_score_diff')
        return df

    def create_age_features(self, df):
        df = df.copy()
        age_cols = ['cookie_age', 'card_age', 'phone_age', 'password_age', 'puid_age']
        existing_age = [c for c in age_cols if c in df.columns]
        if len(existing_age) >= 2:
            df['min_age'] = df[existing_age].min(axis=1)
            df['max_age'] = df[existing_age].max(axis=1)
            df['mean_age'] = df[existing_age].mean(axis=1)
            df['age_range'] = df['max_age'] - df['min_age']
            df['age_std'] = df[existing_age].std(axis=1)
            self.feature_names_created.extend(['min_age', 'max_age', 'mean_age', 'age_range', 'age_std'])
        for col in existing_age:
            flag_name = f'{col}_is_new'
            if str(df[col].dtype) in ('float64', 'int64', 'Int64'):
                threshold = df[col].quantile(0.05)
                df[flag_name] = (df[col] <= threshold).astype(int)
                self.feature_names_created.append(flag_name)
        return df

    def create_velocity_features(self, df):
        df = df.copy()
        if 'puid_orders_1h_without_refunds' in df.columns and 'puid_orders_24h_without_refunds' in df.columns:
            df['velocity_1h_to_24h'] = df['puid_orders_1h_without_refunds'] / (df['puid_orders_24h_without_refunds'] + 1e-08)
            self.feature_names_created.append('velocity_1h_to_24h')
        if 'puid_orders_24h_without_refunds' in df.columns and 'puid_orders_2d_without_refunds' in df.columns:
            df['velocity_24h_to_2d'] = df['puid_orders_24h_without_refunds'] / (df['puid_orders_2d_without_refunds'] + 1e-08)
            self.feature_names_created.append('velocity_24h_to_2d')
        if 'loan_1h_base_limit' in df.columns and 'loan_24h_base_limit' in df.columns:
            df['loan_velocity_1h_24h'] = df['loan_1h_base_limit'] / (df['loan_24h_base_limit'] + 1e-08)
            self.feature_names_created.append('loan_velocity_1h_24h')
        return df

    def create_anomaly_flags(self, df):
        df = df.copy()
        if 'first_order_flg' in df.columns and 'order_loan' in df.columns:
            high_threshold = df['order_loan'].quantile(0.9)
            df['first_order_high_amount'] = ((df['first_order_flg'] == 1) & (df['order_loan'] > high_threshold)).astype(int)
            self.feature_names_created.append('first_order_high_amount')
        if 'phone_diff' in df.columns:
            df['phone_mismatch'] = (df['phone_diff'] != 0).astype(int)
            self.feature_names_created.append('phone_mismatch')
        debt_cols = [c for c in df.columns if c.startswith('debt_on_')]
        if len(debt_cols) >= 2:
            df['multi_debt_flag'] = ((df[debt_cols] > 0).sum(axis=1) >= 3).astype(int)
            self.feature_names_created.append('multi_debt_flag')
        if 'is_night' in df.columns and 'min_age' in df.columns:
            df['night_new_account'] = ((df.get('is_night', 0) == 1) & (df['min_age'] < df['min_age'].quantile(0.1))).astype(int)
            self.feature_names_created.append('night_new_account')
        return df

    def fit_transform(self, df):
        print('  Парсинг временных признаков...')
        df = self.parse_datetime_features(df)
        print('  Создание отношений (ratio features)...')
        df = self.create_ratio_features(df)
        print('  Создание возрастных признаков...')
        df = self.create_age_features(df)
        print('  Создание velocity-признаков...')
        df = self.create_velocity_features(df)
        print('  Создание флагов аномалий...')
        df = self.create_anomaly_flags(df)
        print(f'\n  Всего создано новых признаков: {len(self.feature_names_created)}')
        return df
fe = FeatureEngineer()
df_fe = fe.fit_transform(df)
print(f'\nРазмер после инженерии: {df_fe.shape}')

class DataPreparer:

    def __init__(self, target_col='resolution_fraud', time_col='order_creation_dttm', test_size=0.2, val_size=0.15):
        self.target_col = target_col
        self.time_col = time_col
        self.test_size = test_size
        self.val_size = val_size
        self.scaler = RobustScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.cat_columns = []
        self.num_columns = []

    def identify_columns(self, df):
        exclude_cols = [self.target_col, 'order_id', 'max_iso_eventtime_str', 'order_creation_dttm', 'split_type']
        self.cat_columns = []
        self.num_columns = []
        for col in df.columns:
            if col in exclude_cols:
                continue
            if df[col].dtype == 'object' or (df[col].nunique() < 20 and df[col].dtype in ['int64', 'float64'] and col.endswith('_flg')):
                if df[col].dtype == 'object':
                    self.cat_columns.append(col)
                else:
                    self.num_columns.append(col)
            elif df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                self.num_columns.append(col)
        self.feature_columns = self.num_columns + self.cat_columns
        print(f'  Числовых признаков: {len(self.num_columns)}')
        print(f'  Категориальных признаков: {len(self.cat_columns)}')
        print(f'  Всего признаков: {len(self.feature_columns)}')

    def encode_categoricals(self, df):
        df = df.copy()
        for col in self.cat_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = df[col].fillna('__MISSING__').astype(str)
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        return df

    def temporal_split(self, df):
        if self.time_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[self.time_col]):
            df_sorted = df.sort_values(self.time_col).reset_index(drop=True)
            n = len(df_sorted)
            train_end = int(n * (1 - self.test_size - self.val_size))
            val_end = int(n * (1 - self.test_size))
            df_train = df_sorted.iloc[:train_end]
            df_val = df_sorted.iloc[train_end:val_end]
            df_test = df_sorted.iloc[val_end:]
            print('  Хронологический сплит:')
            print(f'    Train: {len(df_train)} ({df_train[self.target_col].mean():.4%} fraud)')
            print(f'    Val:   {len(df_val)} ({df_val[self.target_col].mean():.4%} fraud)')
            print(f'    Test:  {len(df_test)} ({df_test[self.target_col].mean():.4%} fraud)')
            tc = self.time_col
            print(f'    Train period: {df_train[tc].min()} — {df_train[tc].max()}')
            print(f'    Val period:   {df_val[tc].min()} — {df_val[tc].max()}')
            print(f'    Test period:  {df_test[tc].min()} — {df_test[tc].max()}')
            return (df_train, df_val, df_test)
        print('  Временная колонка недоступна, используем стратифицированный сплит.')
        df_train_val, df_test = train_test_split(df, test_size=self.test_size, stratify=df[self.target_col], random_state=42)
        relative_val = self.val_size / (1 - self.test_size)
        df_train, df_val = train_test_split(df_train_val, test_size=relative_val, stratify=df_train_val[self.target_col], random_state=42)
        print('  Стратифицированный сплит:')
        print(f'    Train: {len(df_train)} ({df_train[self.target_col].mean():.4%} fraud)')
        print(f'    Val:   {len(df_val)} ({df_val[self.target_col].mean():.4%} fraud)')
        print(f'    Test:  {len(df_test)} ({df_test[self.target_col].mean():.4%} fraud)')
        return (df_train, df_val, df_test)

    def handle_missing(self, X_train, X_val, X_test):
        imputer = SimpleImputer(strategy='median')
        num_cols_present = [c for c in self.num_columns if c in X_train.columns]
        X_train[num_cols_present] = imputer.fit_transform(X_train[num_cols_present])
        X_val[num_cols_present] = imputer.transform(X_val[num_cols_present])
        X_test[num_cols_present] = imputer.transform(X_test[num_cols_present])
        self.imputer = imputer
        return (X_train, X_val, X_test)

    def prepare(self, df):
        print('  Идентификация столбцов...')
        self.identify_columns(df)
        print('  Кодирование категориальных переменных...')
        df = self.encode_categoricals(df)
        print('  Разделение на train/val/test...')
        df_train, df_val, df_test = self.temporal_split(df)
        feature_cols = [c for c in self.feature_columns if c in df_train.columns]
        self.feature_columns = feature_cols
        X_train = df_train[feature_cols].copy()
        y_train = df_train[self.target_col].copy()
        X_val = df_val[feature_cols].copy()
        y_val = df_val[self.target_col].copy()
        X_test = df_test[feature_cols].copy()
        y_test = df_test[self.target_col].copy()
        print('  Обработка пропусков...')
        X_train, X_val, X_test = self.handle_missing(X_train, X_val, X_test)
        for frame in (X_train, X_val, X_test):
            frame.replace([np.inf, -np.inf], np.nan, inplace=True)
            frame.fillna(0, inplace=True)
        return (X_train, y_train, X_val, y_val, X_test, y_test)
preparer = DataPreparer(target_col='resolution_fraud', time_col='order_creation_dttm')
X_train, y_train, X_val, y_val, X_test, y_test = preparer.prepare(df_fe)
print(f'\nИтоговые размеры:')
print(f'  X_train: {X_train.shape}, fraud rate: {y_train.mean():.4%}')
print(f'  X_val:   {X_val.shape}, fraud rate: {y_val.mean():.4%}')
print(f'  X_test:  {X_test.shape}, fraud rate: {y_test.mean():.4%}')
