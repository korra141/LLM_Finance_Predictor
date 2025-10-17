"""
Evaluation metrics for financial prediction models.
Includes accuracy, direction accuracy, and financial-specific metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from scipy import stats
import warnings

logger = logging.getLogger(__name__)


class FinancialMetrics:
    """Collection of financial prediction metrics."""
    
    def __init__(self):
        self.metrics_history = []
    
    def calculate_direction_accuracy(
        self,
        y_true: Union[np.ndarray, List],
        y_pred: Union[np.ndarray, List]
    ) -> Dict[str, float]:
        """
        Calculate direction prediction accuracy.
        
        Args:
            y_true: True price directions (0 for down, 1 for up)
            y_pred: Predicted price directions
        
        Returns:
            Dictionary with direction accuracy metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Basic accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Precision, recall, F1
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        # Direction-specific metrics
        up_predictions = y_pred == 1
        down_predictions = y_pred == 0
        
        up_accuracy = accuracy_score(y_true[up_predictions], y_pred[up_predictions]) if up_predictions.any() else 0.0
        down_accuracy = accuracy_score(y_true[down_predictions], y_pred[down_predictions]) if down_predictions.any() else 0.0
        
        return {
            'direction_accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'up_accuracy': up_accuracy,
            'down_accuracy': down_accuracy,
            'total_predictions': len(y_true),
            'up_predictions': up_predictions.sum(),
            'down_predictions': down_predictions.sum()
        }
    
    def calculate_price_metrics(
        self,
        y_true: Union[np.ndarray, List],
        y_pred: Union[np.ndarray, List]
    ) -> Dict[str, float]:
        """
        Calculate price prediction metrics.
        
        Args:
            y_true: True prices
            y_pred: Predicted prices
        
        Returns:
            Dictionary with price prediction metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Standard regression metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Percentage-based metrics
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
        
        # Financial-specific metrics
        price_direction_accuracy = self._calculate_price_direction_accuracy(y_true, y_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mape': mape,
            'smape': smape,
            'price_direction_accuracy': price_direction_accuracy
        }
    
    def _calculate_price_direction_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """Calculate accuracy of price direction predictions."""
        # Calculate actual and predicted price changes
        true_changes = np.diff(y_true)
        pred_changes = np.diff(y_pred)
        
        # Determine directions
        true_directions = (true_changes > 0).astype(int)
        pred_directions = (pred_changes > 0).astype(int)
        
        # Calculate accuracy
        return accuracy_score(true_directions, pred_directions)
    
    def calculate_volatility_metrics(
        self,
        y_true: Union[np.ndarray, List],
        y_pred: Union[np.ndarray, List]
    ) -> Dict[str, float]:
        """
        Calculate volatility prediction metrics.
        
        Args:
            y_true: True volatility values
            y_pred: Predicted volatility values
        
        Returns:
            Dictionary with volatility prediction metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Standard metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Volatility-specific metrics
        volatility_direction_accuracy = self._calculate_volatility_direction_accuracy(y_true, y_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'volatility_direction_accuracy': volatility_direction_accuracy
        }
    
    def _calculate_volatility_direction_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """Calculate accuracy of volatility direction predictions."""
        # Categorize volatility levels
        true_levels = self._categorize_volatility(y_true)
        pred_levels = self._categorize_volatility(y_pred)
        
        return accuracy_score(true_levels, pred_levels)
    
    def _categorize_volatility(self, volatility: np.ndarray) -> np.ndarray:
        """Categorize volatility into low, medium, high."""
        percentiles = np.percentile(volatility, [33, 67])
        categories = np.zeros_like(volatility, dtype=int)
        categories[volatility > percentiles[1]] = 2  # High
        categories[(volatility > percentiles[0]) & (volatility <= percentiles[1])] = 1  # Medium
        return categories
    
    def calculate_sharpe_ratio(
        self,
        returns: Union[np.ndarray, List],
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate Sharpe ratio for a strategy.
        
        Args:
            returns: Strategy returns
            risk_free_rate: Risk-free rate (annual)
        
        Returns:
            Sharpe ratio
        """
        returns = np.array(returns)
        
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
    
    def calculate_max_drawdown(
        self,
        returns: Union[np.ndarray, List]
    ) -> Dict[str, float]:
        """
        Calculate maximum drawdown.
        
        Args:
            returns: Strategy returns
        
        Returns:
            Dictionary with drawdown metrics
        """
        returns = np.array(returns)
        
        # Calculate cumulative returns
        cumulative = np.cumprod(1 + returns)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative)
        
        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max
        
        # Find maximum drawdown
        max_drawdown = np.min(drawdown)
        max_drawdown_idx = np.argmin(drawdown)
        
        # Find recovery point
        recovery_idx = None
        for i in range(max_drawdown_idx, len(cumulative)):
            if cumulative[i] >= running_max[max_drawdown_idx]:
                recovery_idx = i
                break
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_idx': max_drawdown_idx,
            'recovery_idx': recovery_idx,
            'drawdown_duration': recovery_idx - max_drawdown_idx if recovery_idx else None
        }
    
    def calculate_calmar_ratio(
        self,
        returns: Union[np.ndarray, List]
    ) -> float:
        """
        Calculate Calmar ratio.
        
        Args:
            returns: Strategy returns
        
        Returns:
            Calmar ratio
        """
        returns = np.array(returns)
        
        if len(returns) == 0:
            return 0.0
        
        annual_return = np.mean(returns) * 252
        max_dd = self.calculate_max_drawdown(returns)['max_drawdown']
        
        if max_dd == 0:
            return float('inf') if annual_return > 0 else 0.0
        
        return annual_return / abs(max_dd)
    
    def calculate_information_ratio(
        self,
        strategy_returns: Union[np.ndarray, List],
        benchmark_returns: Union[np.ndarray, List]
    ) -> float:
        """
        Calculate information ratio.
        
        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns
        
        Returns:
            Information ratio
        """
        strategy_returns = np.array(strategy_returns)
        benchmark_returns = np.array(benchmark_returns)
        
        if len(strategy_returns) != len(benchmark_returns):
            raise ValueError("Strategy and benchmark returns must have the same length")
        
        # Calculate excess returns
        excess_returns = strategy_returns - benchmark_returns
        
        if len(excess_returns) == 0 or np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def calculate_hit_rate(
        self,
        y_true: Union[np.ndarray, List],
        y_pred: Union[np.ndarray, List],
        threshold: float = 0.0
    ) -> float:
        """
        Calculate hit rate (percentage of correct predictions).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            threshold: Threshold for considering a prediction correct
        
        Returns:
            Hit rate
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if len(y_true) == 0:
            return 0.0
        
        # Calculate absolute errors
        errors = np.abs(y_true - y_pred)
        
        # Count hits (predictions within threshold)
        hits = np.sum(errors <= threshold)
        
        return hits / len(y_true)
    
    def calculate_confidence_metrics(
        self,
        predictions: Union[np.ndarray, List],
        confidences: Union[np.ndarray, List],
        y_true: Union[np.ndarray, List]
    ) -> Dict[str, float]:
        """
        Calculate confidence calibration metrics.
        
        Args:
            predictions: Model predictions
            confidences: Model confidence scores
            y_true: True values
        
        Returns:
            Dictionary with confidence metrics
        """
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        y_true = np.array(y_true)
        
        # Calculate accuracy for different confidence levels
        confidence_bins = np.linspace(0, 1, 11)
        bin_accuracies = []
        bin_confidences = []
        
        for i in range(len(confidence_bins) - 1):
            lower = confidence_bins[i]
            upper = confidence_bins[i + 1]
            
            mask = (confidences >= lower) & (confidences < upper)
            if mask.any():
                bin_accuracy = accuracy_score(y_true[mask], predictions[mask])
                bin_confidence = np.mean(confidences[mask])
                bin_accuracies.append(bin_accuracy)
                bin_confidences.append(bin_confidence)
        
        # Calculate Expected Calibration Error (ECE)
        ece = 0.0
        if bin_accuracies:
            ece = np.mean(np.abs(np.array(bin_accuracies) - np.array(bin_confidences)))
        
        return {
            'expected_calibration_error': ece,
            'average_confidence': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'confidence_range': np.max(confidences) - np.min(confidences)
        }
    
    def calculate_comprehensive_metrics(
        self,
        y_true: Union[np.ndarray, List],
        y_pred: Union[np.ndarray, List],
        task_type: str = 'direction',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics for a given task type.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            task_type: Type of task ('direction', 'price', 'volatility')
            **kwargs: Additional arguments
        
        Returns:
            Dictionary with comprehensive metrics
        """
        if task_type == 'direction':
            return self.calculate_direction_accuracy(y_true, y_pred)
        elif task_type == 'price':
            return self.calculate_price_metrics(y_true, y_pred)
        elif task_type == 'volatility':
            return self.calculate_volatility_metrics(y_true, y_pred)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def log_metrics(self, metrics: Dict[str, Any], step: int = None):
        """Log metrics to history."""
        log_entry = {
            'step': step,
            'timestamp': pd.Timestamp.now(),
            'metrics': metrics
        }
        self.metrics_history.append(log_entry)
    
    def get_metrics_summary(self) -> pd.DataFrame:
        """Get summary of logged metrics."""
        if not self.metrics_history:
            return pd.DataFrame()
        
        summary_data = []
        for entry in self.metrics_history:
            row = {'step': entry['step'], 'timestamp': entry['timestamp']}
            row.update(entry['metrics'])
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)


if __name__ == "__main__":
    # Example usage
    metrics = FinancialMetrics()
    
    # Sample data
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    
    # Calculate direction accuracy
    direction_metrics = metrics.calculate_direction_accuracy(y_true, y_pred)
    print("Direction accuracy metrics:")
    for key, value in direction_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Sample price data
    true_prices = np.random.randn(100).cumsum() + 100
    pred_prices = true_prices + np.random.randn(100) * 0.1
    
    # Calculate price metrics
    price_metrics = metrics.calculate_price_metrics(true_prices, pred_prices)
    print("\nPrice prediction metrics:")
    for key, value in price_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Sample returns
    returns = np.random.randn(100) * 0.02
    
    # Calculate Sharpe ratio
    sharpe = metrics.calculate_sharpe_ratio(returns)
    print(f"\nSharpe ratio: {sharpe:.4f}")
    
    # Calculate max drawdown
    drawdown_metrics = metrics.calculate_max_drawdown(returns)
    print("Max drawdown metrics:")
    for key, value in drawdown_metrics.items():
        print(f"  {key}: {value}")