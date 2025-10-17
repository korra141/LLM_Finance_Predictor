"""
Data preprocessing utilities for financial data.
Handles cleaning, feature engineering, and data transformation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
import ta
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FinancialDataPreprocessor:
    """Preprocess financial data for ML models."""
    
    def __init__(self):
        self.scalers = {}
        self.feature_selectors = {}
    
    def clean_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean price data by handling missing values and outliers.
        
        Args:
            df: Price DataFrame with OHLCV data
        
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Forward fill missing values
        df_clean = df_clean.fillna(method='ffill')
        
        # Remove rows with all NaN values
        df_clean = df_clean.dropna(how='all')
        
        # Handle outliers using IQR method
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        
        logger.info(f"Cleaned price data: {len(df_clean)} records")
        return df_clean
    
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with technical indicators
        """
        df_indicators = df.copy()
        
        # Price-based indicators
        df_indicators['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df_indicators['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df_indicators['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df_indicators['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
        
        # MACD
        df_indicators['MACD'] = ta.trend.macd(df['Close'])
        df_indicators['MACD_signal'] = ta.trend.macd_signal(df['Close'])
        df_indicators['MACD_histogram'] = ta.trend.macd_diff(df['Close'])
        
        # RSI
        df_indicators['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'])
        df_indicators['BB_upper'] = bb.bollinger_hband()
        df_indicators['BB_middle'] = bb.bollinger_mavg()
        df_indicators['BB_lower'] = bb.bollinger_lband()
        df_indicators['BB_width'] = bb.bollinger_wband()
        df_indicators['BB_percent'] = bb.bollinger_pband()
        
        # Volume indicators
        df_indicators['Volume_SMA'] = ta.volume.volume_sma(df['Close'], df['Volume'])
        df_indicators['Volume_ratio'] = df['Volume'] / df_indicators['Volume_SMA']
        
        # Volatility
        df_indicators['Volatility'] = df['Close'].rolling(window=20).std()
        
        # Price changes
        df_indicators['Price_change'] = df['Close'].pct_change()
        df_indicators['Price_change_5d'] = df['Close'].pct_change(periods=5)
        df_indicators['Price_change_20d'] = df['Close'].pct_change(periods=20)
        
        logger.info(f"Created {len([col for col in df_indicators.columns if col not in df.columns])} technical indicators")
        return df_indicators
    
    def create_target_variables(self, df: pd.DataFrame, target_horizons: List[int] = [1, 5, 20]) -> pd.DataFrame:
        """
        Create target variables for different prediction horizons.
        
        Args:
            df: DataFrame with price data
            target_horizons: List of days ahead to predict
        
        Returns:
            DataFrame with target variables
        """
        df_targets = df.copy()
        
        for horizon in target_horizons:
            # Price direction (binary classification)
            df_targets[f'price_direction_{horizon}d'] = (
                df['Close'].shift(-horizon) > df['Close']
            ).astype(int)
            
            # Price change percentage (regression)
            df_targets[f'price_change_{horizon}d'] = (
                df['Close'].shift(-horizon) / df['Close'] - 1
            )
            
            # Volatility prediction
            df_targets[f'volatility_{horizon}d'] = (
                df['Close'].rolling(window=horizon).std().shift(-horizon)
            )
        
        logger.info(f"Created target variables for horizons: {target_horizons}")
        return df_targets
    
    def create_news_features(self, news_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from news data aggregated by date.
        
        Args:
            news_df: DataFrame with news data
            price_df: DataFrame with price data
        
        Returns:
            DataFrame with news features
        """
        # Convert date columns to datetime
        news_df['date'] = pd.to_datetime(news_df['date']).dt.date
        price_df['date'] = pd.to_datetime(price_df.index).date
        
        # Aggregate news by date
        news_agg = news_df.groupby(['date', 'symbol']).agg({
            'title': 'count',  # Number of news articles
            'sentiment': ['mean', 'std', 'count']  # Sentiment statistics
        }).reset_index()
        
        # Flatten column names
        news_agg.columns = ['date', 'symbol', 'news_count', 'sentiment_mean', 'sentiment_std', 'sentiment_count']
        
        # Merge with price data
        price_df_reset = price_df.reset_index()
        price_df_reset['date'] = pd.to_datetime(price_df_reset.index).date
        
        merged_df = price_df_reset.merge(
            news_agg, 
            on=['date', 'symbol'], 
            how='left'
        )
        
        # Fill missing values
        merged_df['news_count'] = merged_df['news_count'].fillna(0)
        merged_df['sentiment_mean'] = merged_df['sentiment_mean'].fillna(0)
        merged_df['sentiment_std'] = merged_df['sentiment_std'].fillna(0)
        
        logger.info(f"Created news features for {len(merged_df)} records")
        return merged_df
    
    def scale_features(self, df: pd.DataFrame, feature_columns: List[str], scaler_type: str = 'standard') -> pd.DataFrame:
        """
        Scale features using specified scaler.
        
        Args:
            df: DataFrame with features
            feature_columns: List of columns to scale
            scaler_type: Type of scaler ('standard' or 'minmax')
        
        Returns:
            DataFrame with scaled features
        """
        df_scaled = df.copy()
        
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'minmax'")
        
        # Fit and transform features
        df_scaled[feature_columns] = scaler.fit_transform(df[feature_columns])
        
        # Store scaler for later use
        self.scalers[scaler_type] = scaler
        
        logger.info(f"Scaled {len(feature_columns)} features using {scaler_type} scaler")
        return df_scaled
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, k: int = 20) -> pd.DataFrame:
        """
        Select top k features using statistical tests.
        
        Args:
            X: Feature DataFrame
            y: Target variable
            k: Number of features to select
        
        Returns:
            DataFrame with selected features
        """
        selector = SelectKBest(score_func=f_regression, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Store selector for later use
        self.feature_selectors['f_regression'] = selector
        
        logger.info(f"Selected {len(selected_features)} features out of {len(X.columns)}")
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def create_sequences(self, df: pd.DataFrame, sequence_length: int = 30, target_column: str = 'Close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            df: DataFrame with time series data
            sequence_length: Length of input sequences
            target_column: Column to use as target
        
        Returns:
            Tuple of (X, y) arrays
        """
        data = df[target_column].values
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created {len(X)} sequences of length {sequence_length}")
        return X, y
    
    def prepare_dataset(self, df: pd.DataFrame, target_column: str, test_size: float = 0.2) -> Dict[str, pd.DataFrame]:
        """
        Prepare dataset for training by splitting into train/test.
        
        Args:
            df: Complete dataset
            target_column: Target variable column
            test_size: Proportion of data for testing
        
        Returns:
            Dictionary with train and test DataFrames
        """
        # Remove rows with missing targets
        df_clean = df.dropna(subset=[target_column])
        
        # Split by time (last test_size% for testing)
        split_idx = int(len(df_clean) * (1 - test_size))
        
        train_df = df_clean.iloc[:split_idx]
        test_df = df_clean.iloc[split_idx:]
        
        logger.info(f"Split dataset: {len(train_df)} train, {len(test_df)} test")
        
        return {
            'train': train_df,
            'test': test_df
        }


if __name__ == "__main__":
    # Example usage
    preprocessor = FinancialDataPreprocessor()
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 105,
        'Low': np.random.randn(100).cumsum() + 95,
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    # Clean data
    clean_data = preprocessor.clean_price_data(sample_data)
    
    # Create technical indicators
    data_with_indicators = preprocessor.create_technical_indicators(clean_data)
    
    # Create target variables
    data_with_targets = preprocessor.create_target_variables(data_with_indicators)
    
    print("Preprocessing completed successfully!")