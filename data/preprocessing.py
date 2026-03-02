"""
Data preprocessing for financial time-series
Prepares OHLC data for anomaly detection
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from ta.trend import MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands


class FinancialTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for financial time-series windows
    """

    def __init__(self, sequences):
        """
        Args:
            sequences: (n_samples, seq_len, n_features) numpy array
        """
        self.sequences = torch.FloatTensor(sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx],


class FinancialDataPreprocessor:
    """
    Preprocess financial OHLC data for anomaly detection
    """

    def __init__(
        self,
        window_size=60,
        stride=1,
        add_technical_indicators=True,
        scaler_type='robust',
        clip_outliers=True,
        outlier_percentile=1.0
    ):
        """
        Args:
            window_size: length of time-series windows
            stride: stride for sliding window
            add_technical_indicators: whether to add technical features
            scaler_type: 'standard', 'robust', or 'minmax'
            clip_outliers: whether to clip extreme outliers before scaling
            outlier_percentile: percentile threshold for clipping (1.0 = 1% and 99%)
        """
        self.window_size = window_size
        self.stride = stride
        self.add_technical_indicators = add_technical_indicators
        self.scaler_type = scaler_type
        self.clip_outliers = clip_outliers
        self.outlier_percentile = outlier_percentile

        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()

        self.feature_names = []
        self.is_fitted = False
        self.clip_bounds = {}  # Store clipping bounds per feature

    def add_features(self, df):
        """
        Add technical indicators as features
        Args:
            df: DataFrame with OHLC columns
        Returns:
            df: DataFrame with added features
        """
        df = df.copy()

        # Price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Moving averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()

        # MACD
        macd = MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        # RSI
        df['rsi'] = RSIIndicator(df['close'], window=14).rsi()

        # Stochastic
        stoch = StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        # ATR
        atr = AverageTrueRange(df['high'], df['low'], df['close'])
        df['atr'] = atr.average_true_range()
        df['atr_pct'] = df['atr'] / df['close']

        # ADX
        adx = ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'] = adx.adx()

        # Bollinger Bands
        bb = BollingerBands(df['close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['close']
        df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])

        # Volume features
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)

        # Price range
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['close_open_range'] = (df['close'] - df['open']) / df['close']

        return df

    def clip_outliers_by_percentile(self, data, fit=True):
        """
        Clip outliers using percentile-based bounds
        Args:
            data: (n_samples, n_features) array
            fit: whether to compute clipping bounds
        Returns:
            clipped_data: data with outliers clipped
        """
        data_clipped = data.copy()

        if fit:
            # Compute clipping bounds for each feature
            for i in range(data.shape[1]):
                lower = np.percentile(data[:, i], self.outlier_percentile)
                upper = np.percentile(data[:, i], 100 - self.outlier_percentile)
                self.clip_bounds[i] = (lower, upper)

        # Apply clipping
        for i in range(data.shape[1]):
            if i in self.clip_bounds:
                lower, upper = self.clip_bounds[i]
                data_clipped[:, i] = np.clip(data_clipped[:, i], lower, upper)

        return data_clipped

    def prepare_data(self, df, fit_scaler=True):
        """
        Prepare raw OHLC data for model input
        Args:
            df: DataFrame with OHLC data
            fit_scaler: whether to fit the scaler
        Returns:
            sequences: (n_samples, window_size, n_features)
            feature_names: list of feature names
        """
        # Ensure required columns
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Add technical indicators if requested
        if self.add_technical_indicators:
            df = self.add_features(df)

        # Track which original row positions survive after dropna
        # Reset index to get integer positions, then track which survive
        df = df.reset_index(drop=True)
        n_before_drop = len(df)
        df = df.dropna()
        # Store surviving integer positions (row numbers in the DataFrame passed to prepare_data)
        self.surviving_indices_ = df.index.values.copy()  # integer positions that survived dropna
        df = df.reset_index(drop=True)

        # Select numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])

        # Remove problematic columns - aggressive filtering
        columns_to_drop = []

        # Check each column for extreme values
        for col in numeric_df.columns:
            col_lower = col.lower()
            col_values = numeric_df[col].values
            col_abs_max = np.abs(col_values).max()

            # Drop if: timestamp-like, unnamed, or has extreme values
            if ('unnamed' in col_lower or 'time' in col_lower or
                'date' in col_lower or 'id' in col_lower):
                columns_to_drop.append(col)
                continue

            # Drop if column has values > 100,000 (likely timestamps or IDs)
            if col_abs_max > 100000:
                columns_to_drop.append(col)
                print(f"⚠️  Dropping '{col}' with extreme values: max={col_abs_max:.2e}")
                continue

            # Drop if column has zero or near-zero variance (not useful)
            if col_values.std() < 1e-10:
                columns_to_drop.append(col)
                print(f"⚠️  Dropping '{col}' with zero variance")
                continue

        if columns_to_drop:
            print(f"✓ Dropping non-feature columns: {columns_to_drop}")
            numeric_df = numeric_df.drop(columns=columns_to_drop)

        if len(numeric_df.columns) == 0:
            raise ValueError("No features remaining after filtering! Check your data.")

        # Store feature names
        self.feature_names = numeric_df.columns.tolist()

        # Convert to numpy
        data = numeric_df.values

        # Clip outliers if requested
        if self.clip_outliers:
            data = self.clip_outliers_by_percentile(data, fit=fit_scaler)
            if fit_scaler:
                print(f"✓ Clipped outliers at {self.outlier_percentile}% percentile")

        # Scale data
        if fit_scaler:
            scaled_data = self.scaler.fit_transform(data)
            self.is_fitted = True
            print(f"✓ Data scaled: range [{scaled_data.min():.4f}, {scaled_data.max():.4f}], mean {scaled_data.mean():.4f}, std {scaled_data.std():.4f}")
        else:
            if not self.is_fitted:
                raise RuntimeError("Scaler not fitted. Call with fit_scaler=True first.")
            scaled_data = self.scaler.transform(data)

        # Create sliding windows
        sequences = self.create_sequences(scaled_data)

        print(f"Created {len(sequences)} sequences of shape {sequences.shape}")
        print(f"Features ({len(self.feature_names)}): {self.feature_names[:10]}...")

        return sequences, self.feature_names

    def create_sequences(self, data):
        """
        Create sliding window sequences
        Args:
            data: (n_timesteps, n_features) array
        Returns:
            sequences: (n_samples, window_size, n_features) array
        """
        sequences = []

        for i in range(0, len(data) - self.window_size + 1, self.stride):
            seq = data[i:i + self.window_size]
            sequences.append(seq)

        return np.array(sequences)

    def inverse_transform(self, scaled_data):
        """
        Inverse transform scaled data back to original scale
        Args:
            scaled_data: scaled data
        Returns:
            original_data: data in original scale
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted")

        return self.scaler.inverse_transform(scaled_data)

    def create_dataloader(self, sequences, batch_size=32, shuffle=True, num_workers=0):
        """
        Create PyTorch DataLoader
        Args:
            sequences: (n_samples, window_size, n_features) array
            batch_size: batch size
            shuffle: whether to shuffle
            num_workers: number of worker processes
        Returns:
            dataloader: PyTorch DataLoader
        """
        dataset = FinancialTimeSeriesDataset(sequences)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )

        return dataloader


def load_forex_data(file_path, start_date=None, end_date=None):
    """
    Load forex OHLC data from CSV
    Args:
        file_path: path to CSV file
        start_date: optional start date filter
        end_date: optional end date filter
    Returns:
        df: DataFrame with OHLC data
    """
    df = pd.read_csv(file_path)

    # Standardize column names
    df.columns = df.columns.str.lower()

    # Parse date/time if available
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], unit='s', errors='coerce')
        df = df.set_index('time')
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.set_index('date')

    # Filter by date range
    if start_date is not None:
        df = df[df.index >= start_date]
    if end_date is not None:
        df = df[df.index <= end_date]

    # Sort by time
    df = df.sort_index()

    print(f"Loaded {len(df)} rows from {file_path}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Columns: {df.columns.tolist()}")

    return df


def split_train_test(sequences, train_ratio=0.8):
    """
    Split sequences into train and test sets (chronological)
    Args:
        sequences: (n_samples, window_size, n_features)
        train_ratio: ratio of training data
    Returns:
        train_sequences, test_sequences
    """
    split_idx = int(len(sequences) * train_ratio)

    train_sequences = sequences[:split_idx]
    test_sequences = sequences[split_idx:]

    print(f"Train: {len(train_sequences)} sequences")
    print(f"Test: {len(test_sequences)} sequences")

    return train_sequences, test_sequences

