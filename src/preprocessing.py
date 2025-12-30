"""
Text preprocessing pipeline for COVID-19 Instagram sentiment analysis.

This module provides comprehensive text cleaning, normalization, and
feature extraction utilities optimized for social media content.

Author: Tharun Ponnam
Email: tharunponnam007@gmail.com
"""

import re
import string
from typing import List, Optional, Dict, Any, Union
import pandas as pd
import numpy as np

# NLTK imports with error handling
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    
    # Download required resources
    for resource in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)
            
except ImportError:
    raise ImportError("NLTK is required. Install with: pip install nltk")


class TextPreprocessor:
    """
    Comprehensive text preprocessing pipeline for social media data.
    
    Handles cleaning, normalization, tokenization, and feature extraction
    for Instagram posts containing emojis, hashtags, and multilingual content.
    
    Attributes
    ----------
    language : str
        Primary language for stopwords
    use_stemming : bool
        Whether to apply stemming
    use_lemmatization : bool
        Whether to apply lemmatization
        
    Examples
    --------
    >>> preprocessor = TextPreprocessor()
    >>> text = "Check out this #COVID19 update! 😷🙏 https://example.com @user"
    >>> clean = preprocessor.preprocess(text)
    >>> print(clean)
    'check covid update'
    """
    
    # Regex patterns compiled for efficiency
    URL_PATTERN = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    MENTION_PATTERN = re.compile(r'@[\w]+')
    HASHTAG_PATTERN = re.compile(r'#(\w+)')
    EMAIL_PATTERN = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
    NUMBER_PATTERN = re.compile(r'\b\d+\b')
    
    # Emoji pattern covering common ranges
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U0001F1E0-\U0001F1FF"  # Flags
        "]+",
        flags=re.UNICODE
    )
    
    # COVID-19 specific terms to preserve
    COVID_TERMS = {
        'covid', 'covid19', 'covid-19', 'coronavirus', 'corona', 
        'pandemic', 'vaccine', 'vaccination', 'lockdown', 'quarantine',
        'social distancing', 'mask', 'ppe', 'ventilator', 'icu',
        'symptoms', 'positive', 'negative', 'test', 'pcr', 'antigen'
    }
    
    def __init__(
        self,
        language: str = 'english',
        use_stemming: bool = False,
        use_lemmatization: bool = True,
        min_token_length: int = 2,
        custom_stopwords: Optional[List[str]] = None
    ):
        """
        Initialize preprocessor.
        
        Parameters
        ----------
        language : str
            Language for stopwords and processing
        use_stemming : bool
            Apply Porter stemming
        use_lemmatization : bool
            Apply WordNet lemmatization
        min_token_length : int
            Minimum token length to keep
        custom_stopwords : list, optional
            Additional stopwords to remove
        """
        self.language = language
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.min_token_length = min_token_length
        
        # Initialize NLTK components
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        # Build stopword set
        self.stop_words = set(stopwords.words(language))
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)
            
        # Remove COVID terms from stopwords
        self.stop_words -= self.COVID_TERMS
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extract features from text before cleaning.
        
        Parameters
        ----------
        text : str
            Raw input text
            
        Returns
        -------
        dict
            Extracted features including emojis, hashtags, mentions
        """
        if pd.isna(text):
            return {
                'emojis': [],
                'hashtags': [],
                'mentions': [],
                'urls': [],
                'emoji_count': 0,
                'hashtag_count': 0,
                'mention_count': 0,
                'url_count': 0,
                'char_count': 0,
                'word_count': 0
            }
            
        text = str(text)
        
        return {
            'emojis': self.EMOJI_PATTERN.findall(text),
            'hashtags': self.HASHTAG_PATTERN.findall(text),
            'mentions': self.MENTION_PATTERN.findall(text),
            'urls': self.URL_PATTERN.findall(text),
            'emoji_count': len(self.EMOJI_PATTERN.findall(text)),
            'hashtag_count': len(self.HASHTAG_PATTERN.findall(text)),
            'mention_count': len(self.MENTION_PATTERN.findall(text)),
            'url_count': len(self.URL_PATTERN.findall(text)),
            'char_count': len(text),
            'word_count': len(text.split())
        }
    
    def clean_text(self, text: str, preserve_hashtag_words: bool = True) -> str:
        """
        Clean and normalize text.
        
        Parameters
        ----------
        text : str
            Raw input text
        preserve_hashtag_words : bool
            Keep hashtag words without the # symbol
            
        Returns
        -------
        str
            Cleaned text
        """
        if pd.isna(text):
            return ""
            
        text = str(text)
        
        # Remove URLs
        text = self.URL_PATTERN.sub('', text)
        
        # Remove email addresses
        text = self.EMAIL_PATTERN.sub('', text)
        
        # Remove mentions
        text = self.MENTION_PATTERN.sub('', text)
        
        # Handle hashtags
        if preserve_hashtag_words:
            text = self.HASHTAG_PATTERN.sub(r'\1', text)
        else:
            text = self.HASHTAG_PATTERN.sub('', text)
        
        # Remove emojis
        text = self.EMOJI_PATTERN.sub('', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove numbers (optional, comment out if numbers are important)
        text = self.NUMBER_PATTERN.sub('', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Parameters
        ----------
        text : str
            Input text
            
        Returns
        -------
        list
            List of tokens
        """
        try:
            return word_tokenize(text)
        except Exception:
            return text.split()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list.
        
        Parameters
        ----------
        tokens : list
            List of tokens
            
        Returns
        -------
        list
            Filtered tokens
        """
        return [t for t in tokens if t.lower() not in self.stop_words]
    
    def apply_stemming(self, tokens: List[str]) -> List[str]:
        """
        Apply Porter stemming to tokens.
        
        Parameters
        ----------
        tokens : list
            List of tokens
            
        Returns
        -------
        list
            Stemmed tokens
        """
        return [self.stemmer.stem(t) for t in tokens]
    
    def apply_lemmatization(self, tokens: List[str]) -> List[str]:
        """
        Apply WordNet lemmatization to tokens.
        
        Parameters
        ----------
        tokens : list
            List of tokens
            
        Returns
        -------
        list
            Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(t) for t in tokens]
    
    def filter_by_length(self, tokens: List[str]) -> List[str]:
        """
        Filter tokens by minimum length.
        
        Parameters
        ----------
        tokens : list
            List of tokens
            
        Returns
        -------
        list
            Filtered tokens
        """
        return [t for t in tokens if len(t) >= self.min_token_length]
    
    def preprocess(
        self, 
        text: str, 
        return_tokens: bool = False
    ) -> Union[str, List[str]]:
        """
        Apply full preprocessing pipeline.
        
        Parameters
        ----------
        text : str
            Raw input text
        return_tokens : bool
            Return token list instead of joined string
            
        Returns
        -------
        str or list
            Preprocessed text or token list
        """
        # Clean text
        cleaned = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(cleaned)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Apply stemming or lemmatization
        if self.use_stemming:
            tokens = self.apply_stemming(tokens)
        elif self.use_lemmatization:
            tokens = self.apply_lemmatization(tokens)
        
        # Filter by length
        tokens = self.filter_by_length(tokens)
        
        if return_tokens:
            return tokens
        return ' '.join(tokens)
    
    def preprocess_batch(
        self, 
        texts: Union[List[str], pd.Series],
        extract_features: bool = True,
        n_jobs: int = 1
    ) -> pd.DataFrame:
        """
        Preprocess a batch of texts.
        
        Parameters
        ----------
        texts : list or pd.Series
            Collection of texts
        extract_features : bool
            Include feature extraction
        n_jobs : int
            Number of parallel jobs (1 for sequential)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with preprocessed texts and features
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
            
        results = []
        
        for text in texts:
            result = {'original_text': text}
            
            if extract_features:
                features = self.extract_features(text)
                result.update(features)
                
            result['cleaned_text'] = self.clean_text(text)
            result['processed_text'] = self.preprocess(text)
            
            results.append(result)
            
        return pd.DataFrame(results)


def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = 'text',
    preprocessor: Optional[TextPreprocessor] = None
) -> pd.DataFrame:
    """
    Preprocess text column in DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    text_column : str
        Name of text column
    preprocessor : TextPreprocessor, optional
        Custom preprocessor instance
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional preprocessing columns
    """
    if preprocessor is None:
        preprocessor = TextPreprocessor()
        
    # Extract features
    features = df[text_column].apply(preprocessor.extract_features)
    features_df = pd.DataFrame(features.tolist())
    
    # Clean and process text
    df['cleaned_text'] = df[text_column].apply(preprocessor.clean_text)
    df['processed_text'] = df[text_column].apply(preprocessor.preprocess)
    
    # Merge features
    for col in features_df.columns:
        if col not in df.columns:
            df[col] = features_df[col]
            
    return df


if __name__ == '__main__':
    # Example usage
    preprocessor = TextPreprocessor()
    
    sample_texts = [
        "Check out this #COVID19 update! 😷🙏 https://example.com @user Stay safe!",
        "Vaccination day! 💉 Finally got my shot. #vaccine #hopeful",
        "This lockdown is so frustrating 😔 When will it end? #quarantine",
        "Healthcare workers are heroes! Thank you 🏥❤️ #frontlineworkers"
    ]
    
    print("Text Preprocessing Examples")
    print("=" * 60)
    
    for text in sample_texts:
        features = preprocessor.extract_features(text)
        cleaned = preprocessor.clean_text(text)
        processed = preprocessor.preprocess(text)
        
        print(f"\nOriginal: {text}")
        print(f"Cleaned:  {cleaned}")
        print(f"Processed: {processed}")
        print(f"Hashtags: {features['hashtags']}")
        print(f"Emojis:   {features['emojis']}")
