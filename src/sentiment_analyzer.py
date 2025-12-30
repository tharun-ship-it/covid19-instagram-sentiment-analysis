"""
Sentiment analysis engine for COVID-19 Instagram posts.

This module provides VADER-based sentiment analysis with support for
multilingual content and social media-specific features.

Author: Tharun Ponnam
Email: tharunponnam007@gmail.com
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging

# NLTK VADER import
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    nltk.download('vader_lexicon', quiet=True)
except ImportError:
    raise ImportError("NLTK is required. Install with: pip install nltk")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Container for sentiment analysis results."""
    text: str
    compound: float
    positive: float
    negative: float
    neutral: float
    label: str
    confidence: float


class SentimentAnalyzer:
    """
    VADER-based sentiment analyzer for social media text.
    
    VADER (Valence Aware Dictionary and sEntiment Reasoner) is specifically
    designed for social media sentiment analysis and handles emojis, 
    emoticons, and informal language effectively.
    
    Attributes
    ----------
    pos_threshold : float
        Compound score threshold for positive classification
    neg_threshold : float
        Compound score threshold for negative classification
    
    Examples
    --------
    >>> analyzer = SentimentAnalyzer()
    >>> result = analyzer.analyze("I love this! Great news! 😊")
    >>> print(f"Sentiment: {result['sentiment']} (score: {result['compound']:.3f})")
    Sentiment: positive (score: 0.802)
    """
    
    def __init__(
        self,
        pos_threshold: float = 0.05,
        neg_threshold: float = -0.05,
        custom_lexicon: Optional[Dict[str, float]] = None
    ):
        """
        Initialize sentiment analyzer.
        
        Parameters
        ----------
        pos_threshold : float
            Compound score >= this value classified as positive
        neg_threshold : float
            Compound score <= this value classified as negative
        custom_lexicon : dict, optional
            Custom word-sentiment mappings to add to VADER
        """
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        
        # Initialize VADER
        self.analyzer = SentimentIntensityAnalyzer()
        
        # Add custom COVID-19 related lexicon
        covid_lexicon = {
            'vaccine': 1.5,
            'vaccinated': 1.8,
            'vaccination': 1.5,
            'recovered': 2.0,
            'recovery': 1.8,
            'hope': 2.0,
            'hopeful': 2.2,
            'grateful': 2.5,
            'safe': 1.5,
            'heroes': 2.5,
            'hero': 2.3,
            'together': 1.5,
            'support': 1.5,
            'pandemic': -1.0,
            'lockdown': -1.5,
            'quarantine': -1.2,
            'isolated': -1.8,
            'isolation': -1.5,
            'death': -3.0,
            'deaths': -3.0,
            'died': -3.0,
            'dying': -3.0,
            'fear': -2.0,
            'scared': -2.0,
            'anxiety': -2.0,
            'anxious': -1.8,
            'crisis': -2.0,
            'overwhelmed': -1.8,
            'exhausted': -1.5,
            'frustrated': -1.8,
            'frustrating': -1.8,
            'positive': 0.5,  # COVID context: positive test
            'negative': 0.3,  # COVID context: negative test
            'symptoms': -1.0,
            'infected': -1.5,
            'spread': -1.0,
            'spreading': -1.2,
            'variant': -1.0,
            'mutation': -1.0,
            'surge': -1.5,
            'wave': -0.8,
        }
        
        # Update VADER lexicon
        self.analyzer.lexicon.update(covid_lexicon)
        
        # Add any custom lexicon
        if custom_lexicon:
            self.analyzer.lexicon.update(custom_lexicon)
    
    def get_sentiment_scores(self, text: str) -> Dict[str, float]:
        """
        Get detailed sentiment scores for text.
        
        Parameters
        ----------
        text : str
            Input text
            
        Returns
        -------
        dict
            Dictionary with neg, neu, pos, and compound scores
        """
        if pd.isna(text) or str(text).strip() == '':
            return {
                'neg': 0.0,
                'neu': 1.0,
                'pos': 0.0,
                'compound': 0.0
            }
        return self.analyzer.polarity_scores(str(text))
    
    def classify_sentiment(self, compound_score: float) -> str:
        """
        Classify sentiment based on compound score.
        
        Parameters
        ----------
        compound_score : float
            VADER compound score (-1 to 1)
            
        Returns
        -------
        str
            Sentiment label: 'positive', 'negative', or 'neutral'
        """
        if compound_score >= self.pos_threshold:
            return 'positive'
        elif compound_score <= self.neg_threshold:
            return 'negative'
        return 'neutral'
    
    def calculate_confidence(self, scores: Dict[str, float]) -> float:
        """
        Calculate confidence score for sentiment prediction.
        
        Based on the clarity of sentiment signal (how dominant
        one sentiment type is over others).
        
        Parameters
        ----------
        scores : dict
            Sentiment scores dictionary
            
        Returns
        -------
        float
            Confidence score (0 to 1)
        """
        compound = abs(scores['compound'])
        
        # Higher absolute compound score = higher confidence
        # Max compound is 1, so this gives 0-1 range
        base_confidence = compound
        
        # Adjust based on sentiment distribution
        max_score = max(scores['pos'], scores['neg'], scores['neu'])
        distribution_factor = max_score
        
        # Combine factors
        confidence = (base_confidence + distribution_factor) / 2
        
        return min(confidence, 1.0)
    
    def analyze(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Complete sentiment analysis for a text.
        
        Parameters
        ----------
        text : str
            Input text
            
        Returns
        -------
        dict
            Dictionary containing:
            - neg, neu, pos: Individual sentiment scores
            - compound: Overall sentiment score (-1 to 1)
            - sentiment: Classification label
            - confidence: Prediction confidence
        """
        scores = self.get_sentiment_scores(text)
        sentiment = self.classify_sentiment(scores['compound'])
        confidence = self.calculate_confidence(scores)
        
        return {
            'neg': scores['neg'],
            'neu': scores['neu'],
            'pos': scores['pos'],
            'compound': scores['compound'],
            'sentiment': sentiment,
            'confidence': confidence
        }
    
    def analyze_detailed(self, text: str) -> SentimentResult:
        """
        Get detailed sentiment analysis as dataclass.
        
        Parameters
        ----------
        text : str
            Input text
            
        Returns
        -------
        SentimentResult
            Dataclass with all sentiment information
        """
        result = self.analyze(text)
        return SentimentResult(
            text=text,
            compound=result['compound'],
            positive=result['pos'],
            negative=result['neg'],
            neutral=result['neu'],
            label=result['sentiment'],
            confidence=result['confidence']
        )
    
    def analyze_batch(
        self, 
        texts: Union[List[str], pd.Series],
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Analyze sentiment for a batch of texts.
        
        Parameters
        ----------
        texts : list or pd.Series
            Collection of text strings
        show_progress : bool
            Show progress indicator
            
        Returns
        -------
        pd.DataFrame
            DataFrame with sentiment scores and classifications
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        results = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            result = self.analyze(text)
            results.append(result)
            
            if show_progress and (i + 1) % 10000 == 0:
                logger.info(f"Processed {i + 1:,}/{total:,} texts")
        
        return pd.DataFrame(results)
    
    def get_sentiment_distribution(
        self, 
        df: pd.DataFrame, 
        sentiment_col: str = 'sentiment'
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate sentiment distribution statistics.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with sentiment results
        sentiment_col : str
            Name of sentiment label column
            
        Returns
        -------
        dict
            Distribution statistics
        """
        counts = df[sentiment_col].value_counts()
        total = len(df)
        
        distribution = {}
        for sentiment in ['positive', 'neutral', 'negative']:
            count = counts.get(sentiment, 0)
            distribution[sentiment] = {
                'count': count,
                'percentage': (count / total) * 100 if total > 0 else 0
            }
            
        return distribution


class MultilingualSentimentAnalyzer(SentimentAnalyzer):
    """
    Extended sentiment analyzer with multilingual support.
    
    Uses language detection and language-specific processing
    for improved accuracy on non-English texts.
    
    Note: For best results on non-English texts, consider using
    language-specific sentiment lexicons or transformer models.
    """
    
    SUPPORTED_LANGUAGES = {'en', 'es', 'pt', 'fr', 'de', 'it'}
    
    def __init__(
        self,
        pos_threshold: float = 0.05,
        neg_threshold: float = -0.05,
        detect_language: bool = True
    ):
        """
        Initialize multilingual analyzer.
        
        Parameters
        ----------
        pos_threshold : float
            Positive classification threshold
        neg_threshold : float
            Negative classification threshold
        detect_language : bool
            Attempt automatic language detection
        """
        super().__init__(pos_threshold, neg_threshold)
        self.detect_language = detect_language
        
        # Try to import langdetect
        try:
            from langdetect import detect, LangDetectException
            self._detect = detect
            self._lang_exception = LangDetectException
        except ImportError:
            logger.warning("langdetect not installed. Language detection disabled.")
            self.detect_language = False
    
    def detect_text_language(self, text: str) -> str:
        """
        Detect language of text.
        
        Parameters
        ----------
        text : str
            Input text
            
        Returns
        -------
        str
            ISO 639-1 language code or 'unknown'
        """
        if not self.detect_language:
            return 'en'
            
        try:
            if pd.isna(text) or len(str(text).strip()) < 10:
                return 'unknown'
            return self._detect(str(text))
        except Exception:
            return 'unknown'
    
    def analyze_with_language(
        self, 
        text: str, 
        language: Optional[str] = None
    ) -> Dict[str, Union[str, float]]:
        """
        Analyze sentiment with language awareness.
        
        Parameters
        ----------
        text : str
            Input text
        language : str, optional
            Language code. If None, will detect.
            
        Returns
        -------
        dict
            Sentiment results with language information
        """
        if language is None:
            language = self.detect_text_language(text)
            
        result = self.analyze(text)
        result['language'] = language
        result['language_supported'] = language in self.SUPPORTED_LANGUAGES
        
        return result


def analyze_dataframe(
    df: pd.DataFrame,
    text_column: str = 'text',
    analyzer: Optional[SentimentAnalyzer] = None
) -> pd.DataFrame:
    """
    Add sentiment analysis to DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    text_column : str
        Name of text column
    analyzer : SentimentAnalyzer, optional
        Custom analyzer instance
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added sentiment columns
    """
    if analyzer is None:
        analyzer = SentimentAnalyzer()
        
    logger.info(f"Analyzing sentiment for {len(df):,} texts...")
    
    # Analyze all texts
    results = analyzer.analyze_batch(df[text_column])
    
    # Add results to DataFrame
    for col in results.columns:
        df[f'sentiment_{col}'] = results[col].values
        
    # Rename for clarity
    df = df.rename(columns={
        'sentiment_sentiment': 'predicted_sentiment',
        'sentiment_compound': 'compound_score'
    })
    
    logger.info("Sentiment analysis complete!")
    return df


if __name__ == '__main__':
    # Example usage
    analyzer = SentimentAnalyzer()
    
    test_texts = [
        "Finally got vaccinated! So grateful and hopeful! 🙏💉",
        "This lockdown is exhausting. When will it end? 😔",
        "COVID-19 update: New guidelines released today.",
        "Healthcare workers are heroes! Thank you for your sacrifice! ❤️",
        "Fear and anxiety every day. This pandemic is devastating."
    ]
    
    print("Sentiment Analysis Examples")
    print("=" * 60)
    
    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"\nText: {text[:50]}...")
        print(f"Sentiment: {result['sentiment']} "
              f"(compound: {result['compound']:.3f}, "
              f"confidence: {result['confidence']:.2f})")
