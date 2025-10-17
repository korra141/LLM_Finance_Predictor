"""
Dataset class for financial prediction tasks.
Handles data loading, preprocessing, and batching for training and inference.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from transformers import AutoTokenizer
import yaml

logger = logging.getLogger(__name__)


class FinancialDataset(Dataset):
    """Dataset class for financial prediction tasks."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: AutoTokenizer,
        retrieval_index: Optional[Any] = None,
        max_length: int = 512,
        target_column: str = 'price_direction_1d',
        include_context: bool = True,
        context_length: int = 5
    ):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.retrieval_index = retrieval_index
        self.max_length = max_length
        self.target_column = target_column
        self.include_context = include_context
        self.context_length = context_length
        
        # Validate data
        self._validate_data()
        
        logger.info(f"Initialized dataset with {len(self.data)} samples")
    
    def _validate_data(self):
        """Validate the input data."""
        required_columns = ['symbol', 'date']
        if self.target_column not in self.data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data")
        
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Check for missing targets
        missing_targets = self.data[self.target_column].isna().sum()
        if missing_targets > 0:
            logger.warning(f"Found {missing_targets} samples with missing targets")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample from the dataset."""
        row = self.data.iloc[idx]
        
        # Prepare features
        features = self._prepare_features(row)
        
        # Prepare context (if retrieval is enabled)
        context = ""
        if self.include_context and self.retrieval_index is not None:
            context = self._get_context(row)
        
        # Prepare text input
        text_input = self._prepare_text_input(row, features, context)
        
        # Tokenize
        encoding = self.tokenizer(
            text_input,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Prepare target
        target = self._prepare_target(row)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': target,
            'symbol': row['symbol'],
            'date': str(row['date'])
        }
    
    def _prepare_features(self, row: pd.Series) -> str:
        """Prepare feature text from row data."""
        features = []
        
        # Price features
        price_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        for feature in price_features:
            if feature in row and pd.notna(row[feature]):
                features.append(f"{feature}: {row[feature]:.2f}")
        
        # Technical indicators
        technical_features = [
            'SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_upper', 'BB_lower',
            'Price_change', 'Price_change_5d', 'Price_change_20d'
        ]
        for feature in technical_features:
            if feature in row and pd.notna(row[feature]):
                features.append(f"{feature}: {row[feature]:.4f}")
        
        # News features
        news_features = ['news_count', 'sentiment_mean', 'sentiment_std']
        for feature in news_features:
            if feature in row and pd.notna(row[feature]):
                features.append(f"{feature}: {row[feature]:.4f}")
        
        return " | ".join(features)
    
    def _get_context(self, row: pd.Series) -> str:
        """Get relevant context using retrieval."""
        try:
            # Create query from current features
            query = f"{row['symbol']} {row['date']} financial data"
            
            # Search for relevant documents
            results = self.retrieval_index.search(query, top_k=self.context_length)
            
            # Combine context
            context_parts = []
            for result in results:
                context_parts.append(result['text'])
            
            return " ".join(context_parts)
        
        except Exception as e:
            logger.warning(f"Error retrieving context: {e}")
            return ""
    
    def _prepare_text_input(self, row: pd.Series, features: str, context: str) -> str:
        """Prepare the complete text input for the model."""
        # Base prompt
        prompt = f"Symbol: {row['symbol']}\nDate: {row['date']}\n"
        
        # Add features
        if features:
            prompt += f"Features: {features}\n"
        
        # Add context
        if context:
            prompt += f"Context: {context}\n"
        
        # Add task instruction
        prompt += "Task: Predict the price direction for the next trading day. "
        prompt += "Respond with 'UP' for price increase or 'DOWN' for price decrease."
        
        return prompt
    
    def _prepare_target(self, row: pd.Series) -> torch.Tensor:
        """Prepare the target tensor."""
        target_value = row[self.target_column]
        
        if pd.isna(target_value):
            # Return a default value for missing targets
            return torch.tensor(0, dtype=torch.long)
        
        # Convert to tensor
        if isinstance(target_value, (int, float)):
            return torch.tensor(int(target_value), dtype=torch.long)
        else:
            # Handle string targets (e.g., 'UP', 'DOWN')
            if target_value.upper() in ['UP', '1', 'TRUE']:
                return torch.tensor(1, dtype=torch.long)
            else:
                return torch.tensor(0, dtype=torch.long)


class FinancialDataModule:
    """Data module for managing datasets and data loaders."""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup_tokenizer(self, model_name: str):
        """Setup the tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Setup tokenizer for model: {model_name}")
    
    def prepare_datasets(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        retrieval_index: Optional[Any] = None
    ):
        """Prepare train, validation, and test datasets."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not setup. Call setup_tokenizer() first.")
        
        # Create datasets
        self.train_dataset = FinancialDataset(
            train_data,
            self.tokenizer,
            retrieval_index=retrieval_index,
            max_length=self.config.get('max_length', 512),
            target_column=self.config.get('target_column', 'price_direction_1d'),
            include_context=self.config.get('include_context', True),
            context_length=self.config.get('context_length', 5)
        )
        
        self.val_dataset = FinancialDataset(
            val_data,
            self.tokenizer,
            retrieval_index=retrieval_index,
            max_length=self.config.get('max_length', 512),
            target_column=self.config.get('target_column', 'price_direction_1d'),
            include_context=self.config.get('include_context', True),
            context_length=self.config.get('context_length', 5)
        )
        
        self.test_dataset = FinancialDataset(
            test_data,
            self.tokenizer,
            retrieval_index=retrieval_index,
            max_length=self.config.get('max_length', 512),
            target_column=self.config.get('target_column', 'price_direction_1d'),
            include_context=self.config.get('include_context', True),
            context_length=self.config.get('context_length', 5)
        )
        
        logger.info(f"Prepared datasets: {len(self.train_dataset)} train, "
                   f"{len(self.val_dataset)} val, {len(self.test_dataset)} test")
    
    def get_data_loaders(
        self,
        batch_size: int = 4,
        num_workers: int = 0,
        shuffle_train: bool = True
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get data loaders for train, validation, and test sets."""
        if any(dataset is None for dataset in [self.train_dataset, self.val_dataset, self.test_dataset]):
            raise ValueError("Datasets not prepared. Call prepare_datasets() first.")
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def get_sample_batch(self, split: str = 'train') -> Dict[str, torch.Tensor]:
        """Get a sample batch for testing."""
        if split == 'train' and self.train_dataset is not None:
            return self.train_dataset[0]
        elif split == 'val' and self.val_dataset is not None:
            return self.val_dataset[0]
        elif split == 'test' and self.test_dataset is not None:
            return self.test_dataset[0]
        else:
            raise ValueError(f"Dataset for split '{split}' not available")


if __name__ == "__main__":
    # Example usage
    from transformers import AutoTokenizer
    
    # Create sample data
    sample_data = pd.DataFrame({
        'symbol': ['AAPL'] * 100,
        'date': pd.date_range('2023-01-01', periods=100),
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 105,
        'Low': np.random.randn(100).cumsum() + 95,
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, 100),
        'SMA_20': np.random.randn(100).cumsum() + 100,
        'RSI': np.random.uniform(20, 80, 100),
        'price_direction_1d': np.random.randint(0, 2, 100)
    })
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
    
    # Create dataset
    dataset = FinancialDataset(
        sample_data,
        tokenizer,
        max_length=256,
        target_column='price_direction_1d'
    )
    
    # Test dataset
    sample = dataset[0]
    print("Sample keys:", sample.keys())
    print("Input shape:", sample['input_ids'].shape)
    print("Target:", sample['labels'])
    
    # Create data loader
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Test data loader
    batch = next(iter(data_loader))
    print("Batch keys:", batch.keys())
    print("Batch input shape:", batch['input_ids'].shape)