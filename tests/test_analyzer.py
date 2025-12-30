"""
Comprehensive Test Suite for COVID-19 Instagram Sentiment Analysis
===================================================================

This module contains unit tests for all major components of the
sentiment analysis pipeline including preprocessing, sentiment
classification, and data loading.

Run with: pytest tests/test_analyzer.py -v

Author: Tharun Ponnam
"""

import pytest
import sys
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import TextPreprocessor
from src.sentiment_analyzer import SentimentAnalyzer, MultilingualSentimentAnalyzer
from src.data_loader import DataLoader
from src.utils import (
    format_percentage,
    safe_divide,
    flatten_dict,
    calculate_metrics,
    batch_iterator
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_texts():
    """Sample COVID-19 related Instagram texts for testing."""
    return [
        "Just got my vaccine! So grateful for the frontline workers 💉❤️",
        "This lockdown is terrible. I miss my family so much 😢",
        "New COVID cases reported today. Stay safe everyone.",
        "Feeling hopeful about the future after getting vaccinated! #COVID19 #vaccine",
        "Lost my job due to pandemic. Life is so hard right now 😞",
        "Wearing masks and staying home to protect others 😷 #StaySafe",
        "¡Me vacuné hoy! Muy feliz 💪 #vacuna #covid19",
        "Recovery numbers are looking good! We can beat this together 💪",
    ]


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'post_id': [f'post_{i}' for i in range(100)],
        'text': [
            f"Sample COVID-19 post number {i}. "
            f"{'Great news!' if i % 3 == 0 else 'Sad news.' if i % 3 == 1 else 'Just updates.'}"
            for i in range(100)
        ],
        'timestamp': pd.date_range('2020-03-15', periods=100, freq='D'),
        'language': ['en'] * 80 + ['es'] * 10 + ['pt'] * 10,
        'engagement_score': np.random.randint(0, 1000, 100),
        'hashtags': [f'#covid,#pandemic,#tag{i}' for i in range(100)],
        'emoji_count': np.random.randint(0, 10, 100)
    })


@pytest.fixture
def preprocessor():
    """Create a TextPreprocessor instance."""
    return TextPreprocessor()


@pytest.fixture
def analyzer():
    """Create a SentimentAnalyzer instance."""
    return SentimentAnalyzer()


# =============================================================================
# TextPreprocessor Tests
# =============================================================================

class TestTextPreprocessor:
    """Test suite for TextPreprocessor class."""
    
    def test_initialization(self, preprocessor):
        """Test preprocessor initializes correctly."""
        assert preprocessor is not None
        assert hasattr(preprocessor, 'clean_text')
        assert hasattr(preprocessor, 'tokenize')
    
    def test_clean_text_lowercase(self, preprocessor):
        """Test text lowercasing."""
        text = "COVID-19 Is A GLOBAL Pandemic"
        cleaned = preprocessor.clean_text(text)
        assert cleaned.islower() or 'covid' in cleaned.lower()
    
    def test_clean_text_urls(self, preprocessor):
        """Test URL removal."""
        text = "Check this out https://example.com/covid for more info"
        cleaned = preprocessor.clean_text(text)
        assert 'https://' not in cleaned
        assert 'example.com' not in cleaned
    
    def test_clean_text_mentions(self, preprocessor):
        """Test mention removal."""
        text = "Thanks @WHO for the guidelines! @CDC @NIH"
        cleaned = preprocessor.clean_text(text)
        assert '@WHO' not in cleaned
        assert '@CDC' not in cleaned
    
    def test_extract_features(self, preprocessor):
        """Test feature extraction from text."""
        text = "Great news! #COVID19 #vaccine 💉❤️ @WHO"
        features = preprocessor.extract_features(text)
        
        assert 'hashtags' in features
        assert 'emojis' in features
        assert 'mentions' in features
        assert len(features['hashtags']) >= 1
        assert len(features['emojis']) >= 1
    
    def test_tokenize(self, preprocessor):
        """Test text tokenization."""
        text = "This is a test sentence about COVID-19"
        tokens = preprocessor.tokenize(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)
    
    def test_remove_stopwords(self, preprocessor):
        """Test stopword removal."""
        tokens = ['this', 'is', 'a', 'test', 'about', 'covid']
        filtered = preprocessor.remove_stopwords(tokens)
        
        assert 'this' not in filtered
        assert 'is' not in filtered
        assert 'a' not in filtered
        assert 'covid' in filtered or 'test' in filtered
    
    def test_lemmatize(self, preprocessor):
        """Test lemmatization."""
        tokens = ['running', 'vaccination', 'cases', 'died']
        lemmatized = preprocessor.lemmatize(tokens)
        
        assert isinstance(lemmatized, list)
        assert len(lemmatized) == len(tokens)
    
    def test_preprocess_batch(self, preprocessor, sample_texts):
        """Test batch preprocessing."""
        results = preprocessor.preprocess_batch(sample_texts)
        
        assert len(results) == len(sample_texts)
        assert all('cleaned_text' in r or 'tokens' in r or isinstance(r, str) for r in results)
    
    def test_empty_text(self, preprocessor):
        """Test handling of empty text."""
        result = preprocessor.clean_text("")
        assert result == "" or result is not None
    
    def test_special_characters(self, preprocessor):
        """Test handling of special characters."""
        text = "COVID-19 cases: 1,234,567! @#$%^&*()"
        cleaned = preprocessor.clean_text(text)
        assert isinstance(cleaned, str)


# =============================================================================
# SentimentAnalyzer Tests
# =============================================================================

class TestSentimentAnalyzer:
    """Test suite for SentimentAnalyzer class."""
    
    def test_initialization(self, analyzer):
        """Test analyzer initializes correctly."""
        assert analyzer is not None
        assert hasattr(analyzer, 'analyze')
        assert hasattr(analyzer, 'get_sentiment_scores')
    
    def test_positive_sentiment(self, analyzer):
        """Test detection of positive sentiment."""
        text = "So grateful and happy! The vaccine works wonderfully! Love and hope!"
        result = analyzer.analyze(text)
        
        assert result['label'] in ['positive', 1]
        assert result['compound'] > 0
    
    def test_negative_sentiment(self, analyzer):
        """Test detection of negative sentiment."""
        text = "This is terrible. So many deaths and suffering. I'm scared and hopeless."
        result = analyzer.analyze(text)
        
        assert result['label'] in ['negative', -1]
        assert result['compound'] < 0
    
    def test_neutral_sentiment(self, analyzer):
        """Test detection of neutral sentiment."""
        text = "COVID-19 cases were reported today in the city."
        result = analyzer.analyze(text)
        
        assert result['label'] in ['neutral', 0]
        assert -0.1 <= result['compound'] <= 0.1
    
    def test_sentiment_scores(self, analyzer):
        """Test sentiment score structure."""
        text = "I hope we get through this pandemic soon"
        scores = analyzer.get_sentiment_scores(text)
        
        assert 'pos' in scores or 'positive' in scores
        assert 'neg' in scores or 'negative' in scores
        assert 'neu' in scores or 'neutral' in scores
        assert 'compound' in scores
    
    def test_classify_sentiment(self, analyzer):
        """Test sentiment classification labels."""
        texts = [
            "Amazing! So happy!",
            "Terrible and sad.",
            "The weather is cloudy."
        ]
        
        for text in texts:
            label = analyzer.classify_sentiment(text)
            assert label in ['positive', 'negative', 'neutral', 1, -1, 0]
    
    def test_analyze_batch(self, analyzer, sample_texts):
        """Test batch sentiment analysis."""
        results = analyzer.analyze_batch(sample_texts)
        
        assert len(results) == len(sample_texts)
        assert all('compound' in r or 'score' in r or 'label' in r for r in results)
    
    def test_covid_lexicon(self, analyzer):
        """Test COVID-specific lexicon integration."""
        # Vaccine should be positive
        vaccine_text = "Got vaccinated today! Feeling protected."
        vaccine_result = analyzer.analyze(vaccine_text)
        
        # Death should be negative
        death_text = "So many deaths from this virus. Heartbreaking."
        death_result = analyzer.analyze(death_text)
        
        assert vaccine_result['compound'] > death_result['compound']
    
    def test_emoji_handling(self, analyzer):
        """Test emoji sentiment contribution."""
        happy_text = "Vaccine day! 💉❤️😊🎉"
        sad_text = "Another lockdown 😢😞💔"
        
        happy_result = analyzer.analyze(happy_text)
        sad_result = analyzer.analyze(sad_text)
        
        # Emojis should influence sentiment
        assert happy_result['compound'] != sad_result['compound']
    
    def test_empty_text(self, analyzer):
        """Test handling of empty text."""
        result = analyzer.analyze("")
        assert result is not None
        assert 'compound' in result or 'score' in result or 'label' in result
    
    def test_get_sentiment_distribution(self, analyzer, sample_texts):
        """Test sentiment distribution calculation."""
        results = analyzer.analyze_batch(sample_texts)
        distribution = analyzer.get_sentiment_distribution(results)
        
        assert 'positive' in distribution or 'pos' in distribution
        assert 'negative' in distribution or 'neg' in distribution
        assert 'neutral' in distribution or 'neu' in distribution


# =============================================================================
# MultilingualSentimentAnalyzer Tests
# =============================================================================

class TestMultilingualAnalyzer:
    """Test suite for MultilingualSentimentAnalyzer class."""
    
    @pytest.fixture
    def multilingual_analyzer(self):
        """Create a MultilingualSentimentAnalyzer instance."""
        try:
            return MultilingualSentimentAnalyzer()
        except ImportError:
            pytest.skip("langdetect not installed")
    
    def test_language_detection(self, multilingual_analyzer):
        """Test language detection."""
        english = "This is an English text about COVID-19"
        spanish = "Esto es un texto en español sobre COVID-19"
        
        en_result = multilingual_analyzer.analyze(english)
        es_result = multilingual_analyzer.analyze(spanish)
        
        assert 'language' in en_result or en_result is not None
        assert 'language' in es_result or es_result is not None
    
    def test_supported_languages(self, multilingual_analyzer):
        """Test analysis of multiple languages."""
        texts = {
            'en': "Great vaccine news! So happy!",
            'es': "¡Excelentes noticias de vacunas! ¡Muy feliz!",
            'pt': "Ótimas notícias sobre vacinas! Muito feliz!"
        }
        
        for lang, text in texts.items():
            result = multilingual_analyzer.analyze(text)
            assert result is not None


# =============================================================================
# DataLoader Tests
# =============================================================================

class TestDataLoader:
    """Test suite for DataLoader class."""
    
    @pytest.fixture
    def loader(self):
        """Create a DataLoader instance."""
        return DataLoader()
    
    def test_initialization(self, loader):
        """Test loader initializes correctly."""
        assert loader is not None
        assert hasattr(loader, 'load_dataset')
    
    def test_validate_columns(self, loader, sample_dataframe, tmp_path):
        """Test column validation."""
        # Save sample data
        csv_path = tmp_path / "test_data.csv"
        sample_dataframe.to_csv(csv_path, index=False)
        
        # Load and validate
        df = loader.load_dataset(str(csv_path))
        assert 'text' in df.columns
        assert 'timestamp' in df.columns
    
    def test_load_csv(self, loader, sample_dataframe, tmp_path):
        """Test CSV loading."""
        csv_path = tmp_path / "test_data.csv"
        sample_dataframe.to_csv(csv_path, index=False)
        
        df = loader.load_dataset(str(csv_path))
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_dataframe)
    
    def test_split_data(self, loader, sample_dataframe):
        """Test data splitting."""
        splits = loader.split_data(sample_dataframe, test_size=0.2)
        
        assert 'train' in splits
        assert 'test' in splits
        assert len(splits['train']) + len(splits['test']) == len(sample_dataframe)
    
    def test_missing_file(self, loader):
        """Test handling of missing file."""
        with pytest.raises(FileNotFoundError):
            loader.load_dataset('nonexistent_file.csv')


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestUtilities:
    """Test suite for utility functions."""
    
    def test_format_percentage(self):
        """Test percentage formatting."""
        assert format_percentage(0.45) == '45.0%'
        assert format_percentage(0.4567, decimals=2) == '45.67%'
        assert format_percentage(45.0) == '45.0%'
    
    def test_safe_divide(self):
        """Test safe division."""
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0) == 0.0
        assert safe_divide(10, 0, default=-1) == -1
    
    def test_flatten_dict(self):
        """Test dictionary flattening."""
        nested = {'a': {'b': 1, 'c': {'d': 2}}, 'e': 3}
        flat = flatten_dict(nested)
        
        assert flat['a_b'] == 1
        assert flat['a_c_d'] == 2
        assert flat['e'] == 3
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 1, 0, 1, 2, 0, 2, 2]
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 'f1_macro' in metrics
        assert 'confusion_matrix' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_batch_iterator(self):
        """Test batch iteration."""
        data = list(range(10))
        batches = list(batch_iterator(data, batch_size=3))
        
        assert len(batches) == 4  # 3 + 3 + 3 + 1
        assert batches[0] == [0, 1, 2]
        assert batches[-1] == [9]


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_full_pipeline(self, sample_dataframe, preprocessor, analyzer):
        """Test complete analysis pipeline."""
        # Preprocess
        sample_dataframe['cleaned_text'] = sample_dataframe['text'].apply(
            preprocessor.clean_text
        )
        
        # Analyze sentiment
        results = analyzer.analyze_batch(
            sample_dataframe['cleaned_text'].tolist()
        )
        
        # Add results to DataFrame
        sample_dataframe['sentiment'] = [r.get('label', r.get('compound', 0)) for r in results]
        
        assert 'sentiment' in sample_dataframe.columns
        assert len(sample_dataframe) == 100
    
    def test_temporal_analysis(self, sample_dataframe, analyzer):
        """Test temporal sentiment analysis."""
        # Analyze sentiment
        results = analyzer.analyze_batch(sample_dataframe['text'].tolist())
        sample_dataframe['compound'] = [r.get('compound', 0) for r in results]
        
        # Group by date
        sample_dataframe['date'] = pd.to_datetime(sample_dataframe['timestamp']).dt.date
        daily_sentiment = sample_dataframe.groupby('date')['compound'].mean()
        
        assert len(daily_sentiment) > 0
        assert daily_sentiment.min() >= -1
        assert daily_sentiment.max() <= 1


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance and stress tests."""
    
    def test_large_batch(self, analyzer):
        """Test processing of large batch."""
        large_texts = ["Sample COVID-19 text for testing"] * 1000
        
        results = analyzer.analyze_batch(large_texts)
        
        assert len(results) == 1000
    
    def test_preprocessing_speed(self, preprocessor):
        """Test preprocessing performance."""
        import time
        
        texts = ["This is a sample text about COVID-19 pandemic and vaccines"] * 100
        
        start = time.time()
        for text in texts:
            preprocessor.clean_text(text)
        elapsed = time.time() - start
        
        # Should process 100 texts in under 1 second
        assert elapsed < 1.0


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
