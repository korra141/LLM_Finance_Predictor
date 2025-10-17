"""
Fine-tuning utilities for financial prediction models.
Handles LoRA, adapters, and other fine-tuning techniques.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import logging
from typing import Dict, Any, Optional, List, Tuple
import yaml
import os
from datetime import datetime
import wandb

logger = logging.getLogger(__name__)


class FinancialModelTrainer:
    """Trainer for fine-tuning financial prediction models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.training_args = None
        
        # Setup logging
        self._setup_logging()
        
        # Initialize model and tokenizer
        self._initialize_model()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = self.config.get('logging', {}).get('log_level', 'info')
        logging.basicConfig(level=getattr(logging, log_level.upper()))
        
        # Setup wandb if enabled
        if self.config.get('logging', {}).get('use_wandb', False):
            wandb.init(
                project=self.config.get('logging', {}).get('wandb_project', 'llm-finance-predictor'),
                config=self.config
            )
    
    def _initialize_model(self):
        """Initialize the base model and tokenizer."""
        model_config = self.config.get('model', {})
        model_name = model_config.get('name', 'microsoft/DialoGPT-medium')
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if model_config.get('quantization') else torch.float32,
            device_map='auto' if model_config.get('device') == 'auto' else None
        )
        
        # Setup LoRA if enabled
        if model_config.get('architecture', {}).get('use_lora', False):
            self._setup_lora()
        
        logger.info(f"Initialized model: {model_name}")
    
    def _setup_lora(self):
        """Setup LoRA configuration."""
        lora_config = self.config.get('model', {}).get('architecture', {})
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config.get('lora_rank', 16),
            lora_alpha=lora_config.get('lora_alpha', 32),
            lora_dropout=lora_config.get('lora_dropout', 0.1),
            target_modules=lora_config.get('target_modules', ["q_proj", "v_proj"]),
            bias="none"
        )
        
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        
        logger.info("Setup LoRA configuration")
    
    def _prepare_training_args(self, output_dir: str) -> TrainingArguments:
        """Prepare training arguments."""
        training_config = self.config.get('training', {})
        validation_config = self.config.get('validation', {})
        logging_config = self.config.get('logging', {})
        
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=training_config.get('num_epochs', 3),
            per_device_train_batch_size=training_config.get('batch_size', 4),
            per_device_eval_batch_size=training_config.get('batch_size', 4),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 4),
            learning_rate=training_config.get('learning_rate', 2e-4),
            weight_decay=training_config.get('weight_decay', 0.01),
            warmup_steps=training_config.get('warmup_steps', 100),
            max_grad_norm=training_config.get('max_grad_norm', 1.0),
            
            # Evaluation and saving
            evaluation_strategy=validation_config.get('eval_strategy', 'steps'),
            eval_steps=validation_config.get('eval_steps', 500),
            save_strategy=validation_config.get('save_strategy', 'steps'),
            save_steps=validation_config.get('save_steps', 1000),
            save_total_limit=validation_config.get('save_total_limit', 3),
            
            # Logging
            logging_steps=logging_config.get('log_steps', 100),
            report_to="wandb" if logging_config.get('use_wandb', False) else None,
            
            # Other settings
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=True,
            remove_unused_columns=False
        )
        
        return self.training_args
    
    def _compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        
        # For classification tasks, compute accuracy
        if len(predictions.shape) == 2:
            predictions = np.argmax(predictions, axis=-1)
        
        # Flatten predictions and labels
        predictions = predictions.flatten()
        labels = labels.flatten()
        
        # Remove padding tokens (-100)
        mask = labels != -100
        predictions = predictions[mask]
        labels = labels[mask]
        
        # Compute accuracy
        accuracy = (predictions == labels).mean()
        
        return {
            'accuracy': accuracy,
            'eval_samples': len(predictions)
        }
    
    def train(
        self,
        train_dataset: DataLoader,
        eval_dataset: Optional[DataLoader] = None,
        output_dir: str = None
    ) -> Trainer:
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            output_dir: Output directory for saving models
        
        Returns:
            Trained trainer object
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"./models/financial_model_{timestamp}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare training arguments
        training_args = self._prepare_training_args(output_dir)
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
            tokenizer=self.tokenizer
        )
        
        # Start training
        logger.info("Starting training...")
        self.trainer.train()
        
        # Save the final model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Training completed. Model saved to {output_dir}")
        
        return self.trainer
    
    def evaluate(self, eval_dataset: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            eval_dataset: Evaluation dataset
        
        Returns:
            Evaluation metrics
        """
        if self.trainer is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        eval_results = self.trainer.evaluate(eval_dataset)
        
        logger.info(f"Evaluation results: {eval_results}")
        return eval_results
    
    def save_model(self, output_dir: str):
        """
        Save the trained model.
        
        Args:
            output_dir: Directory to save the model
        """
        if self.trainer is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save configuration
        with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"Model saved to {output_dir}")
    
    def load_model(self, model_path: str):
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
        """
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        logger.info(f"Model loaded from {model_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if self.model is None:
            return {"status": "Model not initialized"}
        
        info = {
            "model_name": self.config.get('model', {}).get('name'),
            "model_size": self.config.get('model', {}).get('size'),
            "use_lora": self.config.get('model', {}).get('architecture', {}).get('use_lora', False),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in self.model.parameters())
        }
        
        return info


class FinancialModelEvaluator:
    """Evaluator for financial prediction models."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def evaluate_direction_accuracy(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate direction prediction accuracy.
        
        Args:
            test_loader: Test data loader
        
        Returns:
            Dictionary with accuracy metrics
        """
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                labels = batch['labels'].to(self.model.device)
                
                # Get model predictions
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                # Compare with labels
                mask = labels != -100
                if mask.any():
                    correct_predictions += (predictions[mask] == labels[mask]).sum().item()
                    total_predictions += mask.sum().item()
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return {
            'direction_accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions
        }
    
    def evaluate_confidence_calibration(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate confidence calibration of predictions.
        
        Args:
            test_loader: Test data loader
        
        Returns:
            Dictionary with calibration metrics
        """
        self.model.eval()
        confidences = []
        correct = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                labels = batch['labels'].to(self.model.device)
                
                # Get model predictions with probabilities
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                max_probs, predictions = torch.max(probabilities, dim=-1)
                
                # Store confidence and correctness
                mask = labels != -100
                if mask.any():
                    confidences.extend(max_probs[mask].cpu().numpy())
                    correct.extend((predictions[mask] == labels[mask]).cpu().numpy())
        
        # Calculate calibration metrics
        confidences = np.array(confidences)
        correct = np.array(correct)
        
        # Expected Calibration Error (ECE)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = correct[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return {
            'expected_calibration_error': ece,
            'average_confidence': confidences.mean(),
            'average_accuracy': correct.mean()
        }


if __name__ == "__main__":
    # Example usage
    config = {
        'model': {
            'name': 'microsoft/DialoGPT-medium',
            'size': 'medium',
            'architecture': {
                'use_lora': True,
                'lora_rank': 16,
                'lora_alpha': 32
            }
        },
        'training': {
            'batch_size': 4,
            'num_epochs': 1,
            'learning_rate': 2e-4
        },
        'validation': {
            'eval_steps': 100,
            'save_steps': 100
        },
        'logging': {
            'use_wandb': False,
            'log_level': 'info'
        }
    }
    
    # Initialize trainer
    trainer = FinancialModelTrainer(config)
    
    # Get model info
    info = trainer.get_model_info()
    print("Model info:", info)