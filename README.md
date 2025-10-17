# LLM Finance Predictor

A comprehensive system for fine-tuning Large Language Models (LLMs) to predict financial market movements using time series data, news sentiment, and retrieval-augmented generation (RAG).

## 🎯 Overview

This project implements a complete pipeline for:
- **Financial Data Processing**: Loading, cleaning, and feature engineering of market data
- **Technical Analysis**: Calculation of technical indicators and market metrics
- **News Integration**: Sentiment analysis and news impact assessment
- **Retrieval-Augmented Generation (RAG)**: Context-aware predictions using relevant financial documents
- **Model Fine-tuning**: LoRA-based fine-tuning of pre-trained language models
- **Prediction & Explanation**: Generating both predictions and reasoning explanations

## 🏗️ Project Structure

```
llm-finance-predictor/
├── data/
│   ├── raw/                   # Raw downloaded files (prices, news, etc.)
│   ├── processed/             # Cleaned / merged / feature-engineered data
│   ├── retrieval_corpus/      # Documents / news / fundamentals for retrieval
│   └── splits/                # Train / val / test splits
│
├── configs/
│   ├── base_model.yaml        # Base model configuration
│   ├── train.yaml             # Training hyperparameters
│   └── retrieval.yaml         # RAG settings
│
├── src/
│   ├── data/                  # Data loading and preprocessing
│   ├── model/                 # Model training and inference
│   ├── utils/                 # Utilities and metrics
│   └── main.py                # Main entry point
│
├── notebooks/
│   ├── data_exploration.ipynb
│   └── quick_finetune_demo.ipynb
│
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
git clone <repository-url>
cd llm-finance-predictor
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Data Exploration

Start with the data exploration notebook to understand the data structure:

```bash
jupyter notebook notebooks/data_exploration.ipynb
```

### 3. Quick Demo

Run the quick fine-tuning demo to see the system in action:

```bash
jupyter notebook notebooks/quick_finetune_demo.ipynb
```

### 4. Training a Model

```bash
python src/main.py --config configs/base_model.yaml --mode train \
    --symbols AAPL GOOGL MSFT --start_date 2023-01-01 --end_date 2023-12-31 \
    --output_dir ./models/my_model
```

### 5. Making Predictions

```bash
python src/main.py --config configs/base_model.yaml --mode inference \
    --model_path ./models/my_model \
    --features '{"symbol": "AAPL", "Close": 150.0, "RSI": 65.0, "MACD": 0.5}'
```

## 📊 Features

### Data Processing
- **Price Data**: OHLCV data with technical indicators (SMA, RSI, MACD, Bollinger Bands)
- **News Data**: Sentiment analysis and news impact assessment
- **Fundamental Data**: Financial metrics and company information
- **Feature Engineering**: Automated creation of predictive features

### Model Architecture
- **Base Models**: Support for various pre-trained language models
- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning
- **RAG Integration**: Retrieval-augmented generation for context-aware predictions
- **Multi-task Learning**: Support for direction, price, and volatility prediction

### Evaluation Metrics
- **Direction Accuracy**: Binary classification accuracy
- **Financial Metrics**: Sharpe ratio, maximum drawdown, Calmar ratio
- **Confidence Calibration**: Expected calibration error and confidence analysis
- **Risk Assessment**: Volatility prediction and risk metrics

## 🔧 Configuration

The system uses YAML configuration files for easy customization:

### Base Model C