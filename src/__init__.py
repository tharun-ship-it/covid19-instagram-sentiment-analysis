"""
COVID-19 Instagram Sentiment Analysis Package
==============================================

A comprehensive toolkit for analyzing sentiment in COVID-19 related
Instagram posts. Supports multilingual analysis of 500K+ posts spanning
5 years of pandemic discourse (2020-2024).

Modules
-------
data_loader
    Multi-format data loading with validation and preprocessing
preprocessing
    Text cleaning, tokenization, and feature extraction for social media
sentiment_analyzer
    VADER-based sentiment analysis with COVID-19 domain adaptation
visualization
    Publication-ready figures and interactive dashboards
utils
    Helper functions for metrics, logging, and configuration

Example
-------
>>> from src import SentimentAnalyzer, DataLoader, TextPreprocessor
>>> 
>>> # Load and analyze data
>>> loader = DataLoader()
>>> df = loader.load_dataset('data/instagram_posts.csv')
>>> 
>>> # Preprocess text
>>> preprocessor = TextPreprocessor()
>>> df['cleaned_text'] = df['text'].apply(preprocessor.clean_text)
>>> 
>>> # Analyze sentiment
>>> analyzer = SentimentAnalyzer()
>>> results = analyzer.analyze_batch(df['cleaned_text'].tolist())

Author
------
Tharun Ponnam <tharunponnam007@gmail.com>

License
-------
MIT License
"""

__version__ = '1.0.0'
__author__ = 'Tharun Ponnam'
__email__ = 'tharunponnam007@gmail.com'

from .data_loader import DataLoader
from .preprocessing import TextPreprocessor
from .sentiment_analyzer import SentimentAnalyzer, MultilingualSentimentAnalyzer
from .visualization import SentimentVisualizer
from .utils import (
    setup_logging,
    load_config,
    calculate_metrics,
    format_percentage,
    timer_decorator
)

__all__ = [
    # Core classes
    'DataLoader',
    'TextPreprocessor',
    'SentimentAnalyzer',
    'MultilingualSentimentAnalyzer',
    'SentimentVisualizer',
    # Utilities
    'setup_logging',
    'load_config',
    'calculate_metrics',
    'format_percentage',
    'timer_decorator',
    # Metadata
    '__version__',
    '__author__',
    '__email__',
]
