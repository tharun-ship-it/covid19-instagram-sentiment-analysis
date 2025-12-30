"""
Data loading utilities for COVID-19 Instagram sentiment analysis.

This module provides functions for loading, validating, and preprocessing
the Instagram COVID-19 discourse dataset from various sources.

Author: Tharun Ponnam
Email: tharunponnam007@gmail.com
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Union, Tuple
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Multi-format data loader for COVID-19 Instagram dataset.
    
    Supports loading from CSV, JSON, and compressed formats with
    automatic encoding detection and validation.
    
    Attributes
    ----------
    data_path : Path
        Base path to data directory
    encoding : str
        Character encoding for text files
    
    Examples
    --------
    >>> loader = DataLoader(data_path='data/')
    >>> df = loader.load_dataset()
    >>> print(f"Loaded {len(df)} records")
    """
    
    SUPPORTED_FORMATS = {'.csv', '.json', '.parquet', '.xlsx'}
    REQUIRED_COLUMNS = ['text']
    OPTIONAL_COLUMNS = ['timestamp', 'language', 'sentiment_label', 
                        'hashtags', 'engagement_score']
    
    def __init__(
        self, 
        data_path: str = 'data/',
        encoding: str = 'utf-8'
    ):
        """
        Initialize DataLoader.
        
        Parameters
        ----------
        data_path : str
            Path to data directory
        encoding : str
            Default encoding for text files
        """
        self.data_path = Path(data_path)
        self.encoding = encoding
        
    def load_dataset(
        self, 
        filename: Optional[str] = None,
        sample_size: Optional[int] = None,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Load Instagram COVID-19 dataset.
        
        Parameters
        ----------
        filename : str, optional
            Specific filename to load. If None, searches for dataset.
        sample_size : int, optional
            Number of records to sample. If None, loads full dataset.
        random_state : int
            Random seed for reproducible sampling
            
        Returns
        -------
        pd.DataFrame
            Loaded and validated DataFrame
            
        Raises
        ------
        FileNotFoundError
            If dataset file is not found
        ValueError
            If required columns are missing
        """
        # Find dataset file
        if filename:
            filepath = self.data_path / filename
        else:
            filepath = self._find_dataset_file()
            
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {filepath}")
            
        logger.info(f"Loading dataset from: {filepath}")
        
        # Load based on file format
        df = self._load_file(filepath)
        
        # Validate required columns
        self._validate_columns(df)
        
        # Sample if requested
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=random_state)
            logger.info(f"Sampled {sample_size:,} records")
            
        # Basic preprocessing
        df = self._preprocess(df)
        
        logger.info(f"Loaded {len(df):,} records with {len(df.columns)} columns")
        return df
    
    def _find_dataset_file(self) -> Path:
        """Search for dataset file in data directory."""
        patterns = [
            '*covid*instagram*.csv',
            '*instagram*covid*.csv',
            '*.csv'
        ]
        
        for pattern in patterns:
            files = list(self.data_path.glob(pattern))
            if files:
                return files[0]
                
        raise FileNotFoundError(
            f"No dataset found in {self.data_path}. "
            "Please download from: https://zenodo.org/records/13896353"
        )
    
    def _load_file(self, filepath: Path) -> pd.DataFrame:
        """Load file based on format."""
        suffix = filepath.suffix.lower()
        
        if suffix == '.csv':
            return self._load_csv(filepath)
        elif suffix == '.json':
            return pd.read_json(filepath)
        elif suffix == '.parquet':
            return pd.read_parquet(filepath)
        elif suffix == '.xlsx':
            return pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported format: {suffix}")
    
    def _load_csv(self, filepath: Path) -> pd.DataFrame:
        """Load CSV with encoding fallback."""
        encodings = [self.encoding, 'utf-8', 'latin-1', 'cp1252']
        
        for enc in encodings:
            try:
                return pd.read_csv(
                    filepath, 
                    encoding=enc,
                    low_memory=False,
                    on_bad_lines='skip'
                )
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
                
        raise ValueError(f"Could not decode file with any supported encoding")
    
    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate required columns exist."""
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        
        if missing:
            # Try to infer text column
            text_candidates = ['text', 'content', 'message', 'caption', 'post_text']
            for col in text_candidates:
                if col in df.columns:
                    df.rename(columns={col: 'text'}, inplace=True)
                    missing.discard('text')
                    break
                    
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic preprocessing steps."""
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['text'], keep='first')
        
        if len(df) < initial_count:
            logger.info(f"Removed {initial_count - len(df):,} duplicate records")
        
        # Handle missing values in text
        df = df.dropna(subset=['text'])
        
        # Convert timestamp if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    @staticmethod
    def split_data(
        df: pd.DataFrame,
        test_size: float = 0.2,
        stratify_col: Optional[str] = None,
        random_state: int = 42
    ) -> Dict[str, Tuple[pd.Series, pd.Series]]:
        """
        Split data into train and test sets.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        test_size : float
            Fraction of data for test set
        stratify_col : str, optional
            Column to use for stratified splitting
        random_state : int
            Random seed
            
        Returns
        -------
        dict
            Dictionary with 'train' and 'test' tuples of (X, y)
        """
        from sklearn.model_selection import train_test_split
        
        stratify = df[stratify_col] if stratify_col else None
        
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=stratify,
            random_state=random_state
        )
        
        return {
            'train': (train_df['text'], train_df.get('sentiment_label')),
            'test': (test_df['text'], test_df.get('sentiment_label'))
        }


def download_dataset(
    output_dir: str = 'data/',
    source: str = 'zenodo'
) -> Path:
    """
    Download COVID-19 Instagram dataset from source.
    
    Parameters
    ----------
    output_dir : str
        Directory to save downloaded file
    source : str
        Data source ('zenodo' or 'ieee')
        
    Returns
    -------
    Path
        Path to downloaded file
    """
    import urllib.request
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    urls = {
        'zenodo': 'https://zenodo.org/records/13896353/files/instagram_covid19_posts.csv'
    }
    
    url = urls.get(source)
    if not url:
        raise ValueError(f"Unknown source: {source}")
        
    filename = output_path / 'instagram_covid19_posts.csv'
    
    logger.info(f"Downloading dataset from {source}...")
    urllib.request.urlretrieve(url, filename)
    logger.info(f"Dataset saved to: {filename}")
    
    return filename


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='COVID-19 Instagram Data Loader')
    parser.add_argument('--download', action='store_true', help='Download dataset')
    parser.add_argument('--data-path', default='data/', help='Data directory')
    parser.add_argument('--sample', type=int, help='Sample size')
    
    args = parser.parse_args()
    
    if args.download:
        download_dataset(args.data_path)
    else:
        loader = DataLoader(data_path=args.data_path)
        df = loader.load_dataset(sample_size=args.sample)
        print(f"\nDataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
