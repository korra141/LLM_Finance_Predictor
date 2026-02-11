#!/usr/bin/env python
# coding: utf-8

# # Quick Fine-tuning Demo for LLM Finance Predictor
# 
# This notebook demonstrates how to quickly fine-tune a language model for financial prediction tasks.
# 
# ## Overview
# - Load and prepare financial data
# - Setup model and training configuration
# - Fine-tune the model with LoRA
# - Evaluate model performance
# - Make predictions
# 

# In[1]:


# Import necessary libraries
import sys
import os


import pandas as pd
import numpy as np
import torch
import yaml
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


print("Libraries imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")


if torch.cuda.is_available():
    print("🚀 Running on GPU environment")
else:
    print("🐌 Running on CPU environment")
    # Optional: Force Trainer to behave on CPU
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


# In[2]:


# Import our custom modules
sys.path.append('../src')
from data.loader import FinancialDataLoader
from data.preprocess import FinancialDataPreprocessor
from data.dataset import FinancialDataModule
from model.finetune import FinancialModelTrainer, FinancialModelEvaluator
from model.inference import FinancialPredictionPipeline
from utils.logging import FinancialLogger
from utils.metrics import FinancialMetrics
from utils.prompt_templates import FinancialPromptBuilder


# ## 1. Configuration Setup
# 
# Let's create a configuration for our quick demo.
# 

# In[3]:


# Create demo configuration
demo_config = {
    'model': {
        'name': 'microsoft/DialoGPT-small',  # Using small model for demo
        'size': 'small',
        'quantization': None,
        'max_length': 256,
        'temperature': 0.7,
        'top_p': 0.9,
        'top_k': 50,
        'architecture': {
            'use_lora': True,
            'lora_rank': 8,  # Smaller rank for demo
            'lora_alpha': 16,
            'lora_dropout': 0.1,
            'target_modules': ["c_attn", "c_proj"]
        }
    },
    'training': {
        'batch_size': 4,  # Small batch for demo
        'gradient_accumulation_steps': 2,
        'learning_rate': 5e-4,
        'num_epochs': 1,  # Single epoch for demo
        'warmup_steps': 5,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0
    },
    'validation': {
        'eval_steps': 50,
        'save_steps': 100,
        'eval_strategy': 'steps',
        'save_strategy': 'steps',
        'save_total_limit': 2
    },
    'logging': {
        'log_level': 'info',
        'use_wandb': True,  # Disabled for demo
        'log_steps': 1
    },
    'use_retrieval': True,  # Disabled for demo
    'target_column': 'price_direction_1d',
    'max_length': 256,
    'include_context': False
}

print("Demo configuration created:")
print(f"Model: {demo_config['model']['name']}")
print(f"LoRA enabled: {demo_config['model']['architecture']['use_lora']}")
print(f"Training epochs: {demo_config['training']['num_epochs']}")
print(f"Batch size: {demo_config['training']['batch_size']}")


# ## 2. Prepare Sample Data
# 
# Let's create some sample financial data for the demo.
# 

# In[4]:


# Create sample financial data
np.random.seed(42)
n_samples = 100
dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')

# Generate sample price data
base_price = 100
price_changes = np.random.randn(n_samples) * 0.02
prices = [base_price]
for change in price_changes[1:]:
    prices.append(prices[-1] * (1 + change))

# Create sample DataFrame
sample_data = pd.DataFrame({
    'symbol': ['AAPL'] * n_samples,
    'date': dates,
    'Open': [p * (1 + np.random.randn() * 0.01) for p in prices],
    'High': [p * (1 + abs(np.random.randn()) * 0.01) for p in prices],
    'Low': [p * (1 - abs(np.random.randn()) * 0.01) for p in prices],
    'Close': prices,
    'Volume': np.random.randint(1000000, 10000000, n_samples),
    'SMA_20': pd.Series(prices).rolling(20).mean(),
    'RSI': np.random.uniform(20, 80, n_samples),
    'MACD': np.random.randn(n_samples) * 0.5,
    'news_count': np.random.randint(0, 5, n_samples),
    'sentiment_mean': np.random.uniform(-1, 1, n_samples)
})

# Create target variable (price direction for next day)
sample_data['price_direction_1d'] = (sample_data['Close'].shift(-1) > sample_data['Close']).astype(str)
sample_data = sample_data.dropna()

print(f"Sample data created:")
print(f"Shape: {sample_data.shape}")
print(f"Date range: {sample_data['date'].min()} to {sample_data['date'].max()}")
print(f"Target distribution: {sample_data['price_direction_1d'].value_counts().to_dict()}")

# Show sample
print(f"\nSample data:")
print(sample_data[['date', 'Close', 'SMA_20', 'RSI', 'price_direction_1d']].head())


# ## 3. Setup Data Module
# 
# Let's prepare the data for training.
# 

# In[5]:


# Split data into train/val/test
train_size = int(0.7 * len(sample_data))
val_size = int(0.15 * len(sample_data))

train_data = sample_data[:train_size]
val_data = sample_data[train_size:train_size + val_size]
test_data = sample_data[train_size + val_size:]

print(f"Data split:")
print(f"Train: {len(train_data)} samples")
print(f"Validation: {len(val_data)} samples")
print(f"Test: {len(test_data)} samples")

# Initialize data module
data_module = FinancialDataModule('../configs/base_model.yaml')

# Setup tokenizer
model_name = demo_config['model']['name']
data_module.setup_tokenizer(model_name)

# Prepare datasets
data_module.prepare_datasets(
    train_data, val_data, test_data,
    retrieval_index=None  # No retrieval for demo
)

# Get data loaders


train_loader, val_loader, test_loader = data_module.get_data_loaders(
    batch_size=demo_config['training']['batch_size'],
    num_workers=0,  # No multiprocessing for demo
    shuffle_train=True
)

print(f"\nData loaders created:")
print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# Test a sample batch
sample_batch = next(iter(train_loader))
print(f"\nSample batch:")
print(f"Input IDs shape: {sample_batch['input_ids'].shape}")
print(f"Attention mask shape: {sample_batch['attention_mask'].shape}")
print(f"Labels shape: {sample_batch['labels'].shape}")


# ## 4. Initialize Model Trainer
# 
# Let's setup the model trainer with our configuration.
# 

# In[6]:


# Initialize trainer
trainer = FinancialModelTrainer(demo_config)

# Get model info
model_info = trainer.get_model_info()
print("Model information:")
for key, value in model_info.items():
    print(f"  {key}: {value}")

# Check if LoRA is properly applied
if hasattr(trainer.model, 'print_trainable_parameters'):
    print(f"\nTrainable parameters:")
    trainer.model.print_trainable_parameters()


# ## 5. Train the Model
# 
# Let's start the training process.
# 

# In[7]:


import os
os.environ["WANDB_DISABLED"] = "false"

#output_dir = f"./models/demo_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#os.makedirs(output_dir, exist_ok=True)
print(f"Starting training...")
#print(f"Output directory: {output_dir}")


# In[ ]:

trainer.model.gradient_checkpointing_enable()
# # Train the model
trained_trainer = trainer.train(
    train_dataset=data_module.train_dataset,
    eval_dataset=data_module.val_dataset,
   #output_dir=output_dir
)

print("Training completed!")


# ## 6. Evaluate the Model
# 
# Let's evaluate the trained model on the test set.
# 

# In[ ]:


#trainer.load_model(output_dir)


# In[ ]:


# sample = next(iter(test_loader))
# input_ids = sample['input_ids'].to(trainer.model.device)
# attention_mask = sample['attention_mask'].to(trainer.model.device)
# labels = sample['labels'].to(trainer.model.device)

# outputs = trainer.model(input_ids=input_ids, attention_mask=attention_mask)


# In[ ]:


# Initialize evaluator
evaluator = FinancialModelEvaluator(trainer.model, trainer.tokenizer)

print("Starting Evalu1ation")

# Evaluate direction accuracy
eval_results = evaluator.evaluate_direction_accuracy(test_loader)

print("Evaluation results:")
for key, value in eval_results.items():
    print(f"  {key}: {value:.4f}")

# Calculate additional metrics
metrics = FinancialMetrics()

# Get predictions for detailed analysis
trainer.model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(trainer.model.device)
        attention_mask = batch['attention_mask'].to(trainer.model.device)
        labels = batch['labels'].to(trainer.model.device)
        
        outputs = trainer.model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)
        
        mask = labels != -100
        if mask.any():
            all_predictions.extend(predictions[mask].cpu().numpy())
            all_labels.extend(labels[mask].cpu().numpy())

# Calculate comprehensive metrics
comprehensive_metrics = metrics.calculate_direction_accuracy(all_labels, all_predictions)

print(f"\nComprehensive metrics:")
for key, value in comprehensive_metrics.items():
    print(f"  {key}: {value:.4f}")


# ## 7. Make Predictions
# 
# Let's test the model with some sample predictions.
# 

# In[ ]:


# Initialize prediction pipeline
pipeline = FinancialPredictionPipeline(output_dir, None)

# Create sample features for prediction
sample_features = {
    'symbol': 'AAPL',
    'date': '2023-12-01',
    'Close': 150.0,
    'SMA_20': 148.5,
    'RSI': 65.0,
    'MACD': 0.5,
    'news_count': 3,
    'sentiment_mean': 0.2
}

# Make prediction
result = pipeline.run_prediction_pipeline(sample_features, include_explanation=True)

print("Prediction result:")
print(f"Direction: {result['prediction']['prediction']['direction']}")
print(f"Confidence: {result['prediction']['prediction']['confidence']}")
print(f"Explanation: {result['prediction']['prediction']['explanation']}")

# Test with different features
sample_features2 = {
    'symbol': 'AAPL',
    'date': '2023-12-02',
    'Close': 145.0,
    'SMA_20': 150.0,
    'RSI': 35.0,
    'MACD': -0.3,
    'news_count': 1,
    'sentiment_mean': -0.5
}

result2 = pipeline.run_prediction_pipeline(sample_features2, include_explanation=True)

print(f"\nSecond prediction:")
print(f"Direction: {result2['prediction']['prediction']['direction']}")
print(f"Confidence: {result2['prediction']['prediction']['confidence']}")
print(f"Explanation: {result2['prediction']['prediction']['explanation']}")


# ## 8. Batch Predictions
# 
# Let's test batch predictions on the test set.
# 

# In[ ]:


# Prepare features for batch prediction
feature_columns = [col for col in test_data.columns if col not in ['symbol', 'date', 'price_direction_1d']]
features_list = []

for _, row in test_data.iterrows():
    features = {col: row[col] for col in feature_columns}
    features_list.append(features)

print(f"Prepared {len(features_list)} samples for batch prediction")

# Make batch predictions
batch_results = pipeline.run_batch_pipeline(features_list, include_explanations=False)

# Analyze results
predictions = [result['prediction']['prediction']['direction'] for result in batch_results]
true_labels = test_data['price_direction_1d'].values

# Calculate accuracy
correct = sum(1 for pred, true in zip(predictions, true_labels) 
              if (pred == 'UP' and true == 1) or (pred == 'DOWN' and true == 0))
accuracy = correct / len(predictions)

print(f"\nBatch prediction results:")
print(f"Total predictions: {len(predictions)}")
print(f"Correct predictions: {correct}")
print(f"Accuracy: {accuracy:.4f}")

# Show some examples
print(f"\nSample predictions:")
for i in range(min(5, len(predictions))):
    print(f"Sample {i+1}: Predicted {predictions[i]}, Actual {true_labels[i]}")


# ## 9. Summary and Next Steps
# 
# Let's summarize what we've accomplished and discuss next steps.
# 

# In[ ]:


print("=== QUICK FINE-TUNING DEMO SUMMARY ===")
print(f"Model: {demo_config['model']['name']}")
print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
print(f"Test samples: {len(test_data)}")
print(f"Training epochs: {demo_config['training']['num_epochs']}")
print(f"LoRA enabled: {demo_config['model']['architecture']['use_lora']}")

print(f"\nPerformance:")
print(f"Direction accuracy: {eval_results['direction_accuracy']:.4f}")
print(f"Batch prediction accuracy: {accuracy:.4f}")

print(f"\nModel saved to: {output_dir}")

print(f"\n=== NEXT STEPS ===")
print("1. Use more data for better performance")
print("2. Experiment with different model architectures")
print("3. Add retrieval-augmented generation (RAG)")
print("4. Implement more sophisticated prompt engineering")
print("5. Add more evaluation metrics")
print("6. Deploy the model for real-time predictions")

print(f"\n=== DEMO COMPLETED SUCCESSFULLY ===")
print("The LLM Finance Predictor has been fine-tuned and is ready for use!")

