"""
Utility Functions for COVID-19 Instagram Sentiment Analysis
============================================================

This module provides helper functions for logging, configuration,
metrics calculation, and common operations used throughout the
sentiment analysis pipeline.

Functions
---------
setup_logging
    Configure logging with file and console handlers
load_config
    Load YAML configuration files
calculate_metrics
    Compute evaluation metrics for sentiment classification
format_percentage
    Format float values as percentage strings
timer_decorator
    Decorator for timing function execution
flatten_dict
    Flatten nested dictionaries
safe_divide
    Division with zero handling

Author: Tharun Ponnam
"""

import os
import sys
import time
import logging
import functools
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Union
from collections import Counter

import numpy as np
import pandas as pd


def setup_logging(
    log_level: str = 'INFO',
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging with console and optional file output.
    
    Parameters
    ----------
    log_level : str, default='INFO'
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : str, optional
        Path to log file. If None, logs only to console
    log_format : str, optional
        Custom log format string
    
    Returns
    -------
    logging.Logger
        Configured logger instance
    
    Example
    -------
    >>> logger = setup_logging('DEBUG', 'logs/analysis.log')
    >>> logger.info('Starting analysis...')
    """
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logger
    logger = logging.getLogger('covid_sentiment')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to YAML configuration file
    
    Returns
    -------
    dict
        Configuration dictionary
    
    Raises
    ------
    FileNotFoundError
        If configuration file doesn't exist
    
    Example
    -------
    >>> config = load_config('config/config.yaml')
    >>> print(config['data']['source'])
    """
    import yaml
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def calculate_metrics(
    y_true: List[int],
    y_pred: List[int],
    labels: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate classification metrics for sentiment analysis.
    
    Parameters
    ----------
    y_true : list of int
        Ground truth labels
    y_pred : list of int
        Predicted labels
    labels : list of str, optional
        Label names for reporting
    
    Returns
    -------
    dict
        Dictionary containing accuracy, precision, recall, f1, and per-class metrics
    
    Example
    -------
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_pred = [0, 1, 1, 0, 1, 2]
    >>> metrics = calculate_metrics(y_true, y_pred)
    >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report
    )
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'support': len(y_true)
    }
    
    # Per-class metrics
    unique_labels = sorted(set(y_true) | set(y_pred))
    if labels is None:
        labels = [str(l) for l in unique_labels]
    
    for i, label in enumerate(unique_labels):
        if i < len(labels):
            label_name = labels[i]
            mask = y_true == label
            if mask.sum() > 0:
                metrics[f'{label_name}_precision'] = precision_score(
                    y_true == label, y_pred == label, zero_division=0
                )
                metrics[f'{label_name}_recall'] = recall_score(
                    y_true == label, y_pred == label, zero_division=0
                )
    
    return metrics


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format a float value as a percentage string.
    
    Parameters
    ----------
    value : float
        Value to format (0-1 scale or 0-100 scale)
    decimals : int, default=1
        Number of decimal places
    
    Returns
    -------
    str
        Formatted percentage string
    
    Example
    -------
    >>> format_percentage(0.4523)
    '45.2%'
    >>> format_percentage(45.23, decimals=2)
    '45.23%'
    """
    # Detect scale (0-1 or 0-100)
    if 0 <= value <= 1:
        value *= 100
    
    return f"{value:.{decimals}f}%"


def timer_decorator(func: Callable) -> Callable:
    """
    Decorator to measure and log function execution time.
    
    Parameters
    ----------
    func : callable
        Function to wrap
    
    Returns
    -------
    callable
        Wrapped function with timing
    
    Example
    -------
    >>> @timer_decorator
    ... def slow_function():
    ...     time.sleep(1)
    ...     return "done"
    >>> result = slow_function()
    # Logs: slow_function executed in 1.00 seconds
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        logger = logging.getLogger('covid_sentiment')
        logger.info(f"{func.__name__} executed in {elapsed_time:.2f} seconds")
        
        return result
    
    return wrapper


def flatten_dict(
    nested_dict: Dict[str, Any],
    parent_key: str = '',
    separator: str = '_'
) -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    
    Parameters
    ----------
    nested_dict : dict
        Dictionary to flatten
    parent_key : str, default=''
        Prefix for keys
    separator : str, default='_'
        Separator between nested keys
    
    Returns
    -------
    dict
        Flattened dictionary
    
    Example
    -------
    >>> nested = {'a': {'b': 1, 'c': 2}, 'd': 3}
    >>> flatten_dict(nested)
    {'a_b': 1, 'a_c': 2, 'd': 3}
    """
    items = []
    
    for key, value in nested_dict.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, separator).items())
        else:
            items.append((new_key, value))
    
    return dict(items)


def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0
) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Parameters
    ----------
    numerator : float
        The dividend
    denominator : float
        The divisor
    default : float, default=0.0
        Value to return if denominator is zero
    
    Returns
    -------
    float
        Result of division or default value
    
    Example
    -------
    >>> safe_divide(10, 2)
    5.0
    >>> safe_divide(10, 0)
    0.0
    """
    if denominator == 0:
        return default
    return numerator / denominator


def get_date_range(
    df: pd.DataFrame,
    date_column: str = 'timestamp'
) -> Dict[str, str]:
    """
    Get the date range of a DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame
    date_column : str, default='timestamp'
        Name of the date column
    
    Returns
    -------
    dict
        Dictionary with 'start', 'end', and 'duration' keys
    """
    dates = pd.to_datetime(df[date_column])
    start = dates.min()
    end = dates.max()
    duration = end - start
    
    return {
        'start': start.strftime('%Y-%m-%d'),
        'end': end.strftime('%Y-%m-%d'),
        'duration_days': duration.days
    }


def get_distribution(
    series: pd.Series,
    normalize: bool = True
) -> Dict[str, float]:
    """
    Calculate the distribution of values in a Series.
    
    Parameters
    ----------
    series : pandas.Series
        Input series
    normalize : bool, default=True
        Whether to return proportions instead of counts
    
    Returns
    -------
    dict
        Distribution dictionary
    """
    counts = series.value_counts(normalize=normalize)
    return counts.to_dict()


def batch_iterator(
    iterable: List[Any],
    batch_size: int = 1000
):
    """
    Iterate over an iterable in batches.
    
    Parameters
    ----------
    iterable : list
        Input list to batch
    batch_size : int, default=1000
        Size of each batch
    
    Yields
    ------
    list
        Batch of items
    
    Example
    -------
    >>> data = list(range(10))
    >>> for batch in batch_iterator(data, batch_size=3):
    ...     print(batch)
    [0, 1, 2]
    [3, 4, 5]
    [6, 7, 8]
    [9]
    """
    length = len(iterable)
    for start_idx in range(0, length, batch_size):
        end_idx = min(start_idx + batch_size, length)
        yield iterable[start_idx:end_idx]


def memory_usage(df: pd.DataFrame) -> Dict[str, str]:
    """
    Calculate memory usage of a DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame
    
    Returns
    -------
    dict
        Memory usage statistics
    """
    total_bytes = df.memory_usage(deep=True).sum()
    
    # Convert to appropriate unit
    if total_bytes < 1024:
        size_str = f"{total_bytes} B"
    elif total_bytes < 1024 ** 2:
        size_str = f"{total_bytes / 1024:.2f} KB"
    elif total_bytes < 1024 ** 3:
        size_str = f"{total_bytes / (1024 ** 2):.2f} MB"
    else:
        size_str = f"{total_bytes / (1024 ** 3):.2f} GB"
    
    return {
        'total_bytes': total_bytes,
        'formatted': size_str,
        'rows': len(df),
        'columns': len(df.columns)
    }


def create_experiment_id() -> str:
    """
    Create a unique experiment identifier based on timestamp.
    
    Returns
    -------
    str
        Experiment ID in format 'exp_YYYYMMDD_HHMMSS'
    """
    return datetime.now().strftime('exp_%Y%m%d_%H%M%S')


def save_results(
    results: Dict[str, Any],
    output_path: str,
    format: str = 'json'
) -> None:
    """
    Save analysis results to file.
    
    Parameters
    ----------
    results : dict
        Results dictionary to save
    output_path : str
        Output file path
    format : str, default='json'
        Output format ('json', 'yaml', 'csv')
    """
    import json
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
    elif format == 'yaml':
        import yaml
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(results, f, default_flow_style=False)
    elif format == 'csv':
        pd.DataFrame([flatten_dict(results)]).to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")


if __name__ == '__main__':
    # Demo usage
    print("COVID-19 Sentiment Analysis Utilities")
    print("=" * 50)
    
    # Test logging
    logger = setup_logging('DEBUG')
    logger.info("Logger configured successfully")
    
    # Test metrics
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 0, 1, 2, 0, 2, 2]
    metrics = calculate_metrics(y_true, y_pred, ['negative', 'neutral', 'positive'])
    print(f"\nAccuracy: {format_percentage(metrics['accuracy'])}")
    print(f"F1 (macro): {format_percentage(metrics['f1_macro'])}")
    
    # Test timer
    @timer_decorator
    def sample_function():
        time.sleep(0.1)
        return "completed"
    
    result = sample_function()
    print(f"\nFunction result: {result}")
    
    # Test flatten
    nested = {'model': {'name': 'svm', 'params': {'C': 1.0}}, 'accuracy': 0.95}
    flat = flatten_dict(nested)
    print(f"\nFlattened: {flat}")
    
    print("\n✓ All utilities working correctly")
