"""
Prompt templates for financial prediction tasks.
Contains various prompt templates for different prediction scenarios.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime


class PromptTemplates:
    """Collection of prompt templates for financial predictions."""
    
    # Base templates
    BASE_TEMPLATE = """
Symbol: {symbol}
Date: {date}
Features: {features}
Context: {context}

Task: {task_instruction}
"""
    
    # Task-specific templates
    DIRECTION_PREDICTION = """
Based on the provided financial data and context, predict the price direction for the next trading day.

Symbol: {symbol}
Date: {date}
Current Price: {current_price}
Technical Indicators: {technical_indicators}
Market Context: {context}

Please respond with:
1. Direction: UP or DOWN
2. Confidence: HIGH, MEDIUM, or LOW
3. Brief explanation of your reasoning

Response:
"""
    
    PRICE_PREDICTION = """
Based on the provided financial data, predict the closing price for the next trading day.

Symbol: {symbol}
Date: {date}
Current Price: {current_price}
Technical Indicators: {technical_indicators}
Market Context: {context}

Please provide:
1. Predicted price: [number]
2. Confidence interval: [lower_bound, upper_bound]
3. Brief explanation

Response:
"""
    
    VOLATILITY_PREDICTION = """
Predict the volatility (price movement range) for the next trading day.

Symbol: {symbol}
Date: {date}
Current Price: {current_price}
Recent Volatility: {recent_volatility}
Market Context: {context}

Please provide:
1. Expected volatility: HIGH, MEDIUM, or LOW
2. Price range: [min_price, max_price]
3. Brief explanation

Response:
"""
    
    NEWS_IMPACT_ANALYSIS = """
Analyze the impact of recent news on the stock price.

Symbol: {symbol}
Date: {date}
Current Price: {current_price}
Recent News: {news_summary}
Market Context: {context}

Please provide:
1. News Impact: POSITIVE, NEGATIVE, or NEUTRAL
2. Expected Price Change: [percentage]
3. Time Horizon: SHORT-TERM, MEDIUM-TERM, or LONG-TERM
4. Brief explanation

Response:
"""
    
    SECTOR_ANALYSIS = """
Analyze the performance of a sector based on market data.

Sector: {sector}
Date: {date}
Sector Performance: {sector_performance}
Key Stocks: {key_stocks}
Market Context: {context}

Please provide:
1. Sector Outlook: BULLISH, BEARISH, or NEUTRAL
2. Key Drivers: [list of main factors]
3. Risk Factors: [list of potential risks]
4. Brief explanation

Response:
"""
    
    EARNINGS_PREDICTION = """
Predict the impact of upcoming earnings announcement.

Symbol: {symbol}
Date: {date}
Current Price: {current_price}
Expected Earnings: {expected_earnings}
Historical Performance: {historical_performance}
Market Context: {context}

Please provide:
1. Earnings Impact: BEAT, MEET, or MISS
2. Expected Price Reaction: [percentage change]
3. Confidence Level: HIGH, MEDIUM, or LOW
4. Brief explanation

Response:
"""
    
    RISK_ASSESSMENT = """
Assess the risk level for a given stock.

Symbol: {symbol}
Date: {date}
Current Price: {current_price}
Financial Metrics: {financial_metrics}
Market Context: {context}

Please provide:
1. Risk Level: LOW, MEDIUM, or HIGH
2. Risk Factors: [list of main risks]
3. Risk Mitigation: [suggested strategies]
4. Brief explanation

Response:
"""
    
    @classmethod
    def get_template(cls, template_name: str) -> str:
        """Get a specific template by name."""
        templates = {
            'direction': cls.DIRECTION_PREDICTION,
            'price': cls.PRICE_PREDICTION,
            'volatility': cls.VOLATILITY_PREDICTION,
            'news_impact': cls.NEWS_IMPACT_ANALYSIS,
            'sector_analysis': cls.SECTOR_ANALYSIS,
            'earnings': cls.EARNINGS_PREDICTION,
            'risk_assessment': cls.RISK_ASSESSMENT
        }
        
        return templates.get(template_name, cls.BASE_TEMPLATE)
    
    @classmethod
    def format_template(
        cls,
        template_name: str,
        **kwargs
    ) -> str:
        """Format a template with provided values."""
        template = cls.get_template(template_name)
        
        # Set default values
        defaults = {
            'symbol': 'UNKNOWN',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'current_price': 'N/A',
            'technical_indicators': 'N/A',
            'context': 'No additional context provided',
            'features': 'N/A',
            'task_instruction': 'Make a financial prediction based on the provided data.'
        }
        
        # Update defaults with provided values
        defaults.update(kwargs)
        
        return template.format(**defaults)


class FinancialPromptBuilder:
    """Builder class for creating financial prediction prompts."""
    
    def __init__(self):
        self.templates = PromptTemplates()
    
    def build_direction_prediction_prompt(
        self,
        symbol: str,
        current_price: float,
        technical_indicators: Dict[str, float],
        context: Optional[str] = None,
        date: Optional[str] = None
    ) -> str:
        """Build a prompt for direction prediction."""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # Format technical indicators
        indicators_text = ", ".join([
            f"{name}: {value:.4f}" for name, value in technical_indicators.items()
        ])
        
        return self.templates.format_template(
            'direction',
            symbol=symbol,
            date=date,
            current_price=current_price,
            technical_indicators=indicators_text,
            context=context or "No additional context provided"
        )
    
    def build_price_prediction_prompt(
        self,
        symbol: str,
        current_price: float,
        technical_indicators: Dict[str, float],
        context: Optional[str] = None,
        date: Optional[str] = None
    ) -> str:
        """Build a prompt for price prediction."""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # Format technical indicators
        indicators_text = ", ".join([
            f"{name}: {value:.4f}" for name, value in technical_indicators.items()
        ])
        
        return self.templates.format_template(
            'price',
            symbol=symbol,
            date=date,
            current_price=current_price,
            technical_indicators=indicators_text,
            context=context or "No additional context provided"
        )
    
    def build_news_impact_prompt(
        self,
        symbol: str,
        current_price: float,
        news_summary: str,
        context: Optional[str] = None,
        date: Optional[str] = None
    ) -> str:
        """Build a prompt for news impact analysis."""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        return self.templates.format_template(
            'news_impact',
            symbol=symbol,
            date=date,
            current_price=current_price,
            news_summary=news_summary,
            context=context or "No additional context provided"
        )
    
    def build_earnings_prediction_prompt(
        self,
        symbol: str,
        current_price: float,
        expected_earnings: Dict[str, Any],
        historical_performance: Dict[str, Any],
        context: Optional[str] = None,
        date: Optional[str] = None
    ) -> str:
        """Build a prompt for earnings prediction."""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # Format expected earnings
        earnings_text = ", ".join([
            f"{key}: {value}" for key, value in expected_earnings.items()
        ])
        
        # Format historical performance
        historical_text = ", ".join([
            f"{key}: {value}" for key, value in historical_performance.items()
        ])
        
        return self.templates.format_template(
            'earnings',
            symbol=symbol,
            date=date,
            current_price=current_price,
            expected_earnings=earnings_text,
            historical_performance=historical_text,
            context=context or "No additional context provided"
        )
    
    def build_risk_assessment_prompt(
        self,
        symbol: str,
        current_price: float,
        financial_metrics: Dict[str, float],
        context: Optional[str] = None,
        date: Optional[str] = None
    ) -> str:
        """Build a prompt for risk assessment."""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # Format financial metrics
        metrics_text = ", ".join([
            f"{name}: {value:.4f}" for name, value in financial_metrics.items()
        ])
        
        return self.templates.format_template(
            'risk_assessment',
            symbol=symbol,
            date=date,
            current_price=current_price,
            financial_metrics=metrics_text,
            context=context or "No additional context provided"
        )
    
    def build_custom_prompt(
        self,
        template_name: str,
        **kwargs
    ) -> str:
        """Build a custom prompt using any template."""
        return self.templates.format_template(template_name, **kwargs)


class PromptOptimizer:
    """Optimize prompts for better model performance."""
    
    def __init__(self):
        self.optimization_techniques = {
            'few_shot': self._add_few_shot_examples,
            'chain_of_thought': self._add_chain_of_thought,
            'role_playing': self._add_role_playing,
            'constraints': self._add_constraints
        }
    
    def optimize_prompt(
        self,
        base_prompt: str,
        optimization_type: str,
        **kwargs
    ) -> str:
        """Optimize a prompt using specified technique."""
        if optimization_type not in self.optimization_techniques:
            raise ValueError(f"Unknown optimization type: {optimization_type}")
        
        return self.optimization_techniques[optimization_type](base_prompt, **kwargs)
    
    def _add_few_shot_examples(self, prompt: str, examples: List[Dict[str, Any]]) -> str:
        """Add few-shot examples to the prompt."""
        examples_text = "\n\nExamples:\n"
        
        for i, example in enumerate(examples, 1):
            examples_text += f"Example {i}:\n"
            examples_text += f"Input: {example['input']}\n"
            examples_text += f"Output: {example['output']}\n\n"
        
        return prompt + examples_text
    
    def _add_chain_of_thought(self, prompt: str, **kwargs) -> str:
        """Add chain-of-thought reasoning to the prompt."""
        cot_instruction = """
Please think step by step:
1. Analyze the technical indicators
2. Consider the market context
3. Evaluate the news impact
4. Make your prediction
5. Provide your reasoning

"""
        return prompt + cot_instruction
    
    def _add_role_playing(self, prompt: str, role: str = "financial analyst") -> str:
        """Add role-playing context to the prompt."""
        role_instruction = f"\n\nYou are an expert {role} with years of experience in financial markets. "
        role_instruction += "Provide your analysis based on your expertise and the data provided.\n"
        
        return role_instruction + prompt
    
    def _add_constraints(self, prompt: str, constraints: List[str]) -> str:
        """Add constraints to the prompt."""
        constraints_text = "\n\nConstraints:\n"
        for constraint in constraints:
            constraints_text += f"- {constraint}\n"
        
        return prompt + constraints_text


if __name__ == "__main__":
    # Example usage
    builder = FinancialPromptBuilder()
    
    # Build direction prediction prompt
    technical_indicators = {
        'SMA_20': 148.5,
        'RSI': 65.0,
        'MACD': 0.5,
        'BB_upper': 155.0,
        'BB_lower': 145.0
    }
    
    prompt = builder.build_direction_prediction_prompt(
        symbol='AAPL',
        current_price=150.0,
        technical_indicators=technical_indicators,
        context="Apple reported strong quarterly earnings with revenue growth of 15%."
    )
    
    print("Generated prompt:")
    print(prompt)
    
    # Optimize prompt
    optimizer = PromptOptimizer()
    optimized_prompt = optimizer.optimize_prompt(
        prompt,
        'chain_of_thought'
    )
    
    print("\nOptimized prompt:")
    print(optimized_prompt)