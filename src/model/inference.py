"""
Inference pipeline for financial prediction models.
Handles prediction, explanation generation, and result interpretation.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
import json
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)


class FinancialPredictor:
    """Financial prediction model for inference."""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.config = {}
        
        # Load configuration
        if config_path:
            self._load_config(config_path)
        
        # Load model and tokenizer
        self._load_model()
    
    def _load_config(self, config_path: str):
        """Load configuration from file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
    
    def _load_model(self):
        """Load the trained model and tokenizer."""
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map='auto' if torch.cuda.is_available() else None
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Loaded model from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict(
        self,
        features: Dict[str, Any],
        context: Optional[str] = None,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> Dict[str, Any]:
        """
        Make a prediction for given features.
        
        Args:
            features: Dictionary of feature values
            context: Optional context string
            max_length: Maximum length of generated response
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
        
        Returns:
            Dictionary with prediction results
        """
        # Prepare input text
        input_text = self._prepare_input(features, context)
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        )
        
        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate prediction
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Parse prediction
        prediction = self._parse_prediction(response)
        
        return {
            'prediction': prediction,
            'response': response,
            'input_text': input_text,
            'confidence': self._calculate_confidence(outputs, inputs),
            'timestamp': datetime.now().isoformat()
        }
    
    def _prepare_input(self, features: Dict[str, Any], context: Optional[str] = None) -> str:
        """Prepare input text from features and context."""
        # Base prompt
        prompt = f"Symbol: {features.get('symbol', 'UNKNOWN')}\n"
        prompt += f"Date: {features.get('date', datetime.now().strftime('%Y-%m-%d'))}\n"
        
        # Add features
        feature_text = []
        for key, value in features.items():
            if key not in ['symbol', 'date'] and value is not None:
                if isinstance(value, (int, float)):
                    feature_text.append(f"{key}: {value:.4f}")
                else:
                    feature_text.append(f"{key}: {value}")
        
        if feature_text:
            prompt += f"Features: {' | '.join(feature_text)}\n"
        
        # Add context
        if context:
            prompt += f"Context: {context}\n"
        
        # Add task instruction
        prompt += "Task: Predict the price direction for the next trading day. "
        prompt += "Respond with 'UP' for price increase or 'DOWN' for price decrease. "
        prompt += "Provide a brief explanation for your prediction."
        
        return prompt
    
    def _parse_prediction(self, response: str) -> Dict[str, Any]:
        """Parse the model response to extract prediction."""
        response_upper = response.upper()
        
        # Determine direction
        if 'UP' in response_upper or 'INCREASE' in response_upper or 'RISE' in response_upper:
            direction = 'UP'
        elif 'DOWN' in response_upper or 'DECREASE' in response_upper or 'FALL' in response_upper:
            direction = 'DOWN'
        else:
            direction = 'UNKNOWN'
        
        # Extract confidence indicators
        confidence_indicators = ['HIGH', 'MEDIUM', 'LOW', 'STRONG', 'WEAK']
        confidence = 'MEDIUM'  # Default
        
        for indicator in confidence_indicators:
            if indicator in response_upper:
                confidence = indicator
                break
        
        return {
            'direction': direction,
            'confidence': confidence,
            'explanation': response.strip()
        }
    
    def _calculate_confidence(self, outputs: torch.Tensor, inputs: Dict[str, torch.Tensor]) -> float:
        """Calculate confidence score from model outputs."""
        try:
            # Get logits for the last token
            with torch.no_grad():
                logits = self.model(**inputs).logits
                last_token_logits = logits[0, -1, :]
                probabilities = torch.softmax(last_token_logits, dim=-1)
                max_prob = torch.max(probabilities).item()
            
            return max_prob
        except:
            return 0.5  # Default confidence
    
    def batch_predict(
        self,
        features_list: List[Dict[str, Any]],
        contexts: Optional[List[str]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Make predictions for a batch of features.
        
        Args:
            features_list: List of feature dictionaries
            contexts: Optional list of context strings
            **kwargs: Additional arguments for predict()
        
        Returns:
            List of prediction results
        """
        predictions = []
        
        for i, features in enumerate(features_list):
            context = contexts[i] if contexts and i < len(contexts) else None
            prediction = self.predict(features, context, **kwargs)
            predictions.append(prediction)
        
        return predictions
    
    def predict_from_dataframe(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        context_column: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Make predictions from a DataFrame.
        
        Args:
            df: Input DataFrame
            feature_columns: List of columns to use as features
            context_column: Optional column containing context
            **kwargs: Additional arguments for predict()
        
        Returns:
            DataFrame with predictions
        """
        results = []
        
        for idx, row in df.iterrows():
            features = {col: row[col] for col in feature_columns}
            context = row[context_column] if context_column else None
            
            prediction = self.predict(features, context, **kwargs)
            results.append(prediction)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Add original data
        for col in feature_columns:
            results_df[f'input_{col}'] = df[col].values
        
        return results_df


class FinancialExplainer:
    """Generate explanations for financial predictions."""
    
    def __init__(self, predictor: FinancialPredictor):
        self.predictor = predictor
    
    def explain_prediction(
        self,
        features: Dict[str, Any],
        context: Optional[str] = None,
        feature_importance: bool = True
    ) -> Dict[str, Any]:
        """
        Generate explanation for a prediction.
        
        Args:
            features: Feature dictionary
            context: Optional context
            feature_importance: Whether to calculate feature importance
        
        Returns:
            Dictionary with explanation
        """
        # Get base prediction
        prediction = self.predictor.predict(features, context)
        
        explanation = {
            'prediction': prediction['prediction'],
            'base_explanation': prediction['response'],
            'feature_analysis': {},
            'context_analysis': {}
        }
        
        # Analyze feature importance
        if feature_importance:
            explanation['feature_analysis'] = self._analyze_feature_importance(features, context)
        
        # Analyze context contribution
        if context:
            explanation['context_analysis'] = self._analyze_context_contribution(features, context)
        
        return explanation
    
    def _analyze_feature_importance(
        self,
        features: Dict[str, Any],
        context: Optional[str] = None
    ) -> Dict[str, float]:
        """Analyze the importance of individual features."""
        base_prediction = self.predictor.predict(features, context)
        base_direction = base_prediction['prediction']['direction']
        
        feature_importance = {}
        
        for feature_name, feature_value in features.items():
            if feature_name in ['symbol', 'date']:
                continue
            
            # Create modified features with zeroed out feature
            modified_features = features.copy()
            modified_features[feature_name] = 0.0
            
            # Get prediction with modified features
            modified_prediction = self.predictor.predict(modified_features, context)
            modified_direction = modified_prediction['prediction']['direction']
            
            # Calculate importance as change in prediction
            importance = 1.0 if base_direction != modified_direction else 0.0
            feature_importance[feature_name] = importance
        
        return feature_importance
    
    def _analyze_context_contribution(
        self,
        features: Dict[str, Any],
        context: str
    ) -> Dict[str, Any]:
        """Analyze the contribution of context to the prediction."""
        # Prediction without context
        prediction_no_context = self.predictor.predict(features, None)
        
        # Prediction with context
        prediction_with_context = self.predictor.predict(features, context)
        
        return {
            'without_context': prediction_no_context['prediction'],
            'with_context': prediction_with_context['prediction'],
            'context_impact': prediction_no_context['prediction']['direction'] != 
                            prediction_with_context['prediction']['direction']
        }


class FinancialPredictionPipeline:
    """Complete pipeline for financial predictions."""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        self.predictor = FinancialPredictor(model_path, config_path)
        self.explainer = FinancialExplainer(self.predictor)
    
    def run_prediction_pipeline(
        self,
        features: Dict[str, Any],
        context: Optional[str] = None,
        include_explanation: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete prediction pipeline.
        
        Args:
            features: Feature dictionary
            context: Optional context
            include_explanation: Whether to include explanation
        
        Returns:
            Complete prediction results
        """
        # Make prediction
        prediction = self.predictor.predict(features, context)
        
        result = {
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add explanation if requested
        if include_explanation:
            explanation = self.explainer.explain_prediction(features, context)
            result['explanation'] = explanation
        
        return result
    
    def run_batch_pipeline(
        self,
        features_list: List[Dict[str, Any]],
        contexts: Optional[List[str]] = None,
        include_explanations: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run the complete prediction pipeline for a batch of inputs.
        
        Args:
            features_list: List of feature dictionaries
            contexts: Optional list of contexts
            include_explanations: Whether to include explanations
        
        Returns:
            List of complete prediction results
        """
        results = []
        
        for i, features in enumerate(features_list):
            context = contexts[i] if contexts and i < len(contexts) else None
            result = self.run_prediction_pipeline(features, context, include_explanations)
            results.append(result)
        
        return results


if __name__ == "__main__":
    # Example usage
    model_path = "./models/financial_model"
    
    # Initialize pipeline
    pipeline = FinancialPredictionPipeline(model_path)
    
    # Example features
    features = {
        'symbol': 'AAPL',
        'date': '2023-12-01',
        'Close': 150.0,
        'SMA_20': 148.5,
        'RSI': 65.0,
        'Volume': 50000000
    }
    
    # Example context
    context = "Apple reported strong quarterly earnings with revenue growth of 15%."
    
    # Run prediction
    result = pipeline.run_prediction_pipeline(features, context)
    
    print("Prediction result:")
    print(json.dumps(result, indent=2))