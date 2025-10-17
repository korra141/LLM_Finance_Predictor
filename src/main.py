"""
Main entry point for the LLM Finance Predictor.
Handles training, evaluation, and inference workflows.
"""

import argparse
import yaml
import logging
import os
import sys
from typing import Dict, Any, Optional
import pandas as pd
import torch
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.loader import FinancialDataLoader
from data.preprocess import FinancialDataPreprocessor
from data.retrieval_index import FinancialRetrievalIndex
from data.dataset import FinancialDataModule
from model.finetune import FinancialModelTrainer, FinancialModelEvaluator
from model.inference import FinancialPredictionPipeline
from utils.logging import FinancialLogger
from utils.metrics import FinancialMetrics
from utils.prompt_templates import FinancialPromptBuilder

logger = logging.getLogger(__name__)


class FinancialPredictorApp:
    """Main application class for the financial predictor."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = None
        self.data_loader = None
        self.preprocessor = None
        self.retrieval_index = None
        self.data_module = None
        self.trainer = None
        self.pipeline = None
        
        # Initialize components
        self._initialize_components()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _initialize_components(self):
        """Initialize all components."""
        # Initialize logger
        self.logger = FinancialLogger(self.config)
        
        # Initialize data components
        self.data_loader = FinancialDataLoader()
        self.preprocessor = FinancialDataPreprocessor()
        
        # Initialize retrieval index if enabled
        if self.config.get('use_retrieval', False):
            retrieval_config = self.config.get('retrieval', {})
            self.retrieval_index = FinancialRetrievalIndex(retrieval_config)
        
        # Initialize data module
        self.data_module = FinancialDataModule(self.config_path)
        
        logger.info("All components initialized successfully")
    
    def prepare_data(self, symbols: list, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for training/evaluation.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for data
            end_date: End date for data
        
        Returns:
            Dictionary with prepared datasets
        """
        logger.info(f"Preparing data for symbols: {symbols}")
        
        # Load raw data
        price_data = self.data_loader.load_price_data(symbols, start_date, end_date)
        news_data = self.data_loader.load_news_data(symbols, start_date, end_date)
        fundamental_data = self.data_loader.load_fundamental_data(symbols)
        
        # Process data
        processed_data = {}
        for symbol, df in price_data.items():
            # Clean data
            clean_df = self.preprocessor.clean_price_data(df)
            
            # Add technical indicators
            df_with_indicators = self.preprocessor.create_technical_indicators(clean_df)
            
            # Add target variables
            df_with_targets = self.preprocessor.create_target_variables(df_with_indicators)
            
            # Add news features
            symbol_news = news_data[news_data['symbol'] == symbol] if not news_data.empty else pd.DataFrame()
            if not symbol_news.empty:
                df_with_news = self.preprocessor.create_news_features(symbol_news, df_with_targets)
            else:
                df_with_news = df_with_targets.copy()
                df_with_news['news_count'] = 0
                df_with_news['sentiment_mean'] = 0
                df_with_news['sentiment_std'] = 0
            
            processed_data[symbol] = df_with_news
        
        # Combine all symbols
        combined_data = pd.concat(processed_data.values(), ignore_index=True)
        
        # Build retrieval index if enabled
        if self.retrieval_index is not None:
            self.retrieval_index.build_from_news_data(news_data)
            self.retrieval_index.build_from_fundamental_data(
                pd.DataFrame(fundamental_data).T.reset_index().rename(columns={'index': 'symbol'})
            )
        
        # Prepare datasets
        target_column = self.config.get('target_column', 'price_direction_1d')
        datasets = self.preprocessor.prepare_dataset(combined_data, target_column)
        
        logger.info(f"Data preparation completed. Train: {len(datasets['train'])}, Test: {len(datasets['test'])}")
        
        return datasets
    
    def train_model(self, train_data: pd.DataFrame, val_data: pd.DataFrame, output_dir: str = None):
        """
        Train the financial prediction model.
        
        Args:
            train_data: Training data
            val_data: Validation data
            output_dir: Output directory for saving model
        """
        logger.info("Starting model training...")
        
        # Setup tokenizer
        model_config = self.config.get('model', {})
        model_name = model_config.get('name', 'microsoft/DialoGPT-medium')
        self.data_module.setup_tokenizer(model_name)
        
        # Prepare datasets
        self.data_module.prepare_datasets(
            train_data, val_data, val_data,  # Using val_data as test for now
            retrieval_index=self.retrieval_index
        )
        
        # Get data loaders
        train_loader, val_loader, test_loader = self.data_module.get_data_loaders(
            batch_size=self.config.get('training', {}).get('batch_size', 4)
        )
        
        # Initialize trainer
        self.trainer = FinancialModelTrainer(self.config)
        
        # Train model
        trained_trainer = self.trainer.train(
            train_dataset=train_loader,
            eval_dataset=val_loader,
            output_dir=output_dir
        )
        
        # Evaluate model
        evaluator = FinancialModelEvaluator(self.trainer.model, self.trainer.tokenizer)
        eval_results = evaluator.evaluate_direction_accuracy(test_loader)
        
        # Log results
        self.logger.log_metrics(eval_results)
        self.logger.log_experiment_summary({
            'final_accuracy': eval_results['direction_accuracy'],
            'training_completed': True
        })
        
        logger.info(f"Training completed. Final accuracy: {eval_results['direction_accuracy']:.4f}")
    
    def evaluate_model(self, test_data: pd.DataFrame, model_path: str):
        """
        Evaluate a trained model.
        
        Args:
            test_data: Test data
            model_path: Path to trained model
        """
        logger.info(f"Evaluating model from {model_path}")
        
        # Initialize pipeline
        self.pipeline = FinancialPredictionPipeline(model_path, self.config_path)
        
        # Prepare test data
        feature_columns = [col for col in test_data.columns if col not in ['symbol', 'date', 'price_direction_1d']]
        
        # Make predictions
        predictions_df = self.pipeline.predictor.predict_from_dataframe(
            test_data, feature_columns
        )
        
        # Calculate metrics
        metrics = FinancialMetrics()
        y_true = test_data['price_direction_1d'].values
        y_pred = [1 if 'UP' in str(pred).upper() else 0 for pred in predictions_df['prediction']]
        
        direction_metrics = metrics.calculate_direction_accuracy(y_true, y_pred)
        
        # Log results
        self.logger.log_metrics(direction_metrics)
        
        logger.info(f"Evaluation completed. Accuracy: {direction_metrics['direction_accuracy']:.4f}")
        
        return direction_metrics
    
    def run_inference(self, features: Dict[str, Any], model_path: str, context: str = None):
        """
        Run inference on a single sample.
        
        Args:
            features: Feature dictionary
            model_path: Path to trained model
            context: Optional context string
        
        Returns:
            Prediction results
        """
        logger.info("Running inference...")
        
        # Initialize pipeline
        self.pipeline = FinancialPredictionPipeline(model_path, self.config_path)
        
        # Make prediction
        result = self.pipeline.run_prediction_pipeline(features, context)
        
        logger.info(f"Inference completed. Prediction: {result['prediction']['prediction']['direction']}")
        
        return result
    
    def run_batch_inference(self, features_list: list, model_path: str, contexts: list = None):
        """
        Run inference on a batch of samples.
        
        Args:
            features_list: List of feature dictionaries
            model_path: Path to trained model
            contexts: Optional list of context strings
        
        Returns:
            List of prediction results
        """
        logger.info(f"Running batch inference on {len(features_list)} samples...")
        
        # Initialize pipeline
        self.pipeline = FinancialPredictionPipeline(model_path, self.config_path)
        
        # Make predictions
        results = self.pipeline.run_batch_pipeline(features_list, contexts)
        
        logger.info("Batch inference completed")
        
        return results
    
    def cleanup(self):
        """Cleanup resources."""
        if self.logger:
            self.logger.finish()
        logger.info("Cleanup completed")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='LLM Finance Predictor')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'inference', 'batch_inference'], 
                       required=True, help='Mode to run')
    parser.add_argument('--symbols', type=str, nargs='+', default=['AAPL', 'GOOGL', 'MSFT'], 
                       help='Stock symbols to use')
    parser.add_argument('--start_date', type=str, default='2023-01-01', help='Start date for data')
    parser.add_argument('--end_date', type=str, default='2023-12-31', help='End date for data')
    parser.add_argument('--model_path', type=str, help='Path to model for evaluation/inference')
    parser.add_argument('--output_dir', type=str, help='Output directory for training')
    parser.add_argument('--features', type=str, help='JSON string of features for inference')
    parser.add_argument('--context', type=str, help='Context string for inference')
    
    args = parser.parse_args()
    
    # Initialize app
    app = FinancialPredictorApp(args.config)
    
    try:
        if args.mode == 'train':
            # Prepare data
            datasets = app.prepare_data(args.symbols, args.start_date, args.end_date)
            
            # Train model
            app.train_model(datasets['train'], datasets['test'], args.output_dir)
            
        elif args.mode == 'eval':
            if not args.model_path:
                raise ValueError("Model path required for evaluation")
            
            # Prepare data
            datasets = app.prepare_data(args.symbols, args.start_date, args.end_date)
            
            # Evaluate model
            app.evaluate_model(datasets['test'], args.model_path)
            
        elif args.mode == 'inference':
            if not args.model_path or not args.features:
                raise ValueError("Model path and features required for inference")
            
            # Parse features
            import json
            features = json.loads(args.features)
            
            # Run inference
            result = app.run_inference(features, args.model_path, args.context)
            print(json.dumps(result, indent=2))
            
        elif args.mode == 'batch_inference':
            if not args.model_path:
                raise ValueError("Model path required for batch inference")
            
            # Prepare data
            datasets = app.prepare_data(args.symbols, args.start_date, args.end_date)
            
            # Prepare features list
            feature_columns = [col for col in datasets['test'].columns if col not in ['symbol', 'date', 'price_direction_1d']]
            features_list = []
            for _, row in datasets['test'].iterrows():
                features = {col: row[col] for col in feature_columns}
                features_list.append(features)
            
            # Run batch inference
            results = app.run_batch_inference(features_list, args.model_path)
            
            # Print results
            for i, result in enumerate(results):
                print(f"Sample {i}: {result['prediction']['prediction']['direction']}")
    
    except Exception as e:
        logger.error(f"Error in {args.mode} mode: {e}")
        raise
    
    finally:
        app.cleanup()


if __name__ == "__main__":
    main()