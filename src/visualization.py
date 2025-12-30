"""
Visualization utilities for COVID-19 Instagram sentiment analysis.

This module provides publication-ready visualizations for sentiment
analysis results, temporal trends, and linguistic patterns.

Author: Tharun Ponnam
Email: tharunponnam007@gmail.com
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from collections import Counter
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Try to import optional dependencies
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    
try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False


class SentimentVisualizer:
    """
    Visualization engine for sentiment analysis results.
    
    Generates publication-ready figures including:
    - Sentiment distribution charts
    - Temporal trend analysis
    - Language comparisons
    - Word clouds
    - Correlation matrices
    
    Attributes
    ----------
    output_dir : Path
        Directory for saving figures
    style : str
        Matplotlib style
    figsize : tuple
        Default figure size
    dpi : int
        Figure resolution
    
    Examples
    --------
    >>> viz = SentimentVisualizer(output_dir='figures/')
    >>> viz.plot_sentiment_distribution(df, 'predicted_sentiment')
    >>> viz.save_figure('sentiment_dist.png')
    """
    
    # Color scheme for sentiment visualization
    SENTIMENT_COLORS = {
        'positive': '#2ecc71',
        'negative': '#e74c3c',
        'neutral': '#95a5a6'
    }
    
    # Color palette for general use
    PALETTE = [
        '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
        '#1abc9c', '#e67e22', '#34495e', '#16a085', '#d35400'
    ]
    
    def __init__(
        self,
        output_dir: str = 'assets/figures/',
        style: str = 'seaborn-v0_8-whitegrid',
        figsize: Tuple[int, int] = (12, 6),
        dpi: int = 150,
        save_format: str = 'png'
    ):
        """
        Initialize visualizer.
        
        Parameters
        ----------
        output_dir : str
            Directory for saving figures
        style : str
            Matplotlib style name
        figsize : tuple
            Default figure dimensions (width, height)
        dpi : int
            Figure resolution
        save_format : str
            Default save format (png, pdf, svg)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.figsize = figsize
        self.dpi = dpi
        self.save_format = save_format
        
        # Set matplotlib style
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use('seaborn-v0_8')
        
        # Configure matplotlib
        plt.rcParams['figure.figsize'] = figsize
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
    
    def save_figure(
        self, 
        filename: str, 
        fig: Optional[plt.Figure] = None,
        tight: bool = True
    ) -> Path:
        """
        Save current or specified figure.
        
        Parameters
        ----------
        filename : str
            Output filename
        fig : plt.Figure, optional
            Figure to save. If None, saves current figure.
        tight : bool
            Use tight bounding box
            
        Returns
        -------
        Path
            Path to saved file
        """
        filepath = self.output_dir / filename
        
        if fig is None:
            fig = plt.gcf()
            
        fig.savefig(
            filepath,
            dpi=self.dpi,
            bbox_inches='tight' if tight else None,
            facecolor='white',
            edgecolor='none'
        )
        
        return filepath
    
    def plot_sentiment_distribution(
        self,
        df: pd.DataFrame,
        sentiment_col: str = 'predicted_sentiment',
        title: str = 'Sentiment Distribution',
        show_values: bool = True
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create pie chart of sentiment distribution.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with sentiment data
        sentiment_col : str
            Column containing sentiment labels
        title : str
            Plot title
        show_values : bool
            Show percentage values on chart
            
        Returns
        -------
        tuple
            (Figure, Axes) objects
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Calculate distribution
        counts = df[sentiment_col].value_counts()
        
        # Ensure consistent order
        order = ['positive', 'neutral', 'negative']
        counts = counts.reindex([s for s in order if s in counts.index])
        
        # Colors
        colors = [self.SENTIMENT_COLORS.get(s, '#cccccc') for s in counts.index]
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=[s.capitalize() for s in counts.index],
            autopct='%1.1f%%' if show_values else None,
            colors=colors,
            explode=[0.02] * len(counts),
            shadow=True,
            startangle=90
        )
        
        # Style percentage text
        for autotext in autotexts:
            autotext.set_fontweight('bold')
            autotext.set_fontsize(11)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        return fig, ax
    
    def plot_compound_histogram(
        self,
        df: pd.DataFrame,
        score_col: str = 'compound_score',
        title: str = 'Distribution of Sentiment Scores',
        bins: int = 50
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create histogram of compound sentiment scores.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with sentiment scores
        score_col : str
            Column containing compound scores
        title : str
            Plot title
        bins : int
            Number of histogram bins
            
        Returns
        -------
        tuple
            (Figure, Axes) objects
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        scores = df[score_col].dropna()
        
        # Create histogram with color gradient
        n, bins_arr, patches = ax.hist(
            scores, bins=bins, alpha=0.7, edgecolor='black', linewidth=0.5
        )
        
        # Color bars by sentiment
        for patch, left_edge in zip(patches, bins_arr[:-1]):
            if left_edge >= 0.05:
                patch.set_facecolor(self.SENTIMENT_COLORS['positive'])
            elif left_edge <= -0.05:
                patch.set_facecolor(self.SENTIMENT_COLORS['negative'])
            else:
                patch.set_facecolor(self.SENTIMENT_COLORS['neutral'])
        
        # Add reference lines
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax.axvline(x=scores.mean(), color='#e74c3c', linestyle='--', linewidth=2,
                   label=f'Mean: {scores.mean():.3f}')
        
        ax.set_xlabel('Compound Score')
        ax.set_ylabel('Frequency')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        
        return fig, ax
    
    def plot_sentiment_timeline(
        self,
        df: pd.DataFrame,
        date_col: str = 'timestamp',
        score_col: str = 'compound_score',
        title: str = 'Sentiment Over Time',
        rolling_window: int = 7,
        show_milestones: bool = True
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create time series plot of sentiment trends.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with temporal sentiment data
        date_col : str
            Column containing datetime
        score_col : str
            Column containing sentiment scores
        title : str
            Plot title
        rolling_window : int
            Days for rolling average
        show_milestones : bool
            Show COVID-19 milestone markers
            
        Returns
        -------
        tuple
            (Figure, Axes) objects
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Ensure datetime
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Daily aggregation
        daily = df.groupby(df[date_col].dt.date)[score_col].agg(['mean', 'std']).reset_index()
        daily.columns = ['date', 'mean', 'std']
        daily['date'] = pd.to_datetime(daily['date'])
        
        # Calculate rolling average
        daily['rolling'] = daily['mean'].rolling(window=rolling_window, min_periods=1).mean()
        
        # Plot
        ax.fill_between(daily['date'], daily['mean'], alpha=0.3, 
                       color='#3498db', label='Daily Mean')
        ax.plot(daily['date'], daily['rolling'], color='#e74c3c', 
               linewidth=2, label=f'{rolling_window}-Day Rolling Avg')
        
        # Reference line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # COVID milestones
        if show_milestones:
            milestones = {
                '2020-03-11': 'WHO Pandemic\nDeclaration',
                '2020-12-11': 'First Vaccine\nApproved',
                '2021-11-26': 'Omicron\nDetected'
            }
            
            for date_str, label in milestones.items():
                try:
                    milestone_date = pd.to_datetime(date_str)
                    if daily['date'].min() <= milestone_date <= daily['date'].max():
                        ax.axvline(x=milestone_date, color='gray', 
                                  linestyle='--', alpha=0.5)
                        ax.annotate(label, xy=(milestone_date, ax.get_ylim()[1] * 0.9),
                                   fontsize=8, ha='center', rotation=0)
                except:
                    pass
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Compound Sentiment Score')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        
        # Format x-axis
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=45)
        
        return fig, ax
    
    def plot_language_comparison(
        self,
        df: pd.DataFrame,
        lang_col: str = 'language',
        score_col: str = 'compound_score',
        title: str = 'Sentiment by Language'
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create comparison of sentiment across languages.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with language and sentiment data
        lang_col : str
            Column containing language codes
        score_col : str
            Column containing sentiment scores
        title : str
            Plot title
            
        Returns
        -------
        tuple
            (Figure, Axes) objects
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Language name mapping
        lang_names = {
            'en': 'English', 'es': 'Spanish', 'pt': 'Portuguese',
            'fr': 'French', 'de': 'German', 'it': 'Italian',
            'other': 'Other'
        }
        
        # Language distribution
        lang_counts = df[lang_col].value_counts()
        
        ax1 = axes[0]
        colors = plt.cm.Set3(np.linspace(0, 1, len(lang_counts)))
        ax1.pie(
            lang_counts.values,
            labels=[lang_names.get(l, l) for l in lang_counts.index],
            autopct='%1.1f%%',
            colors=colors
        )
        ax1.set_title('Language Distribution', fontweight='bold')
        
        # Sentiment by language
        lang_sentiment = df.groupby(lang_col)[score_col].mean().sort_values(ascending=True)
        
        ax2 = axes[1]
        colors = [self.SENTIMENT_COLORS['positive'] if x > 0 
                  else self.SENTIMENT_COLORS['negative'] 
                  for x in lang_sentiment.values]
        
        y_labels = [lang_names.get(l, l) for l in lang_sentiment.index]
        bars = ax2.barh(y_labels, lang_sentiment.values, color=colors, alpha=0.8)
        
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Mean Compound Score')
        ax2.set_title('Sentiment by Language', fontweight='bold')
        
        plt.tight_layout()
        return fig, axes
    
    def plot_wordcloud(
        self,
        texts: Union[List[str], pd.Series],
        title: str = 'Word Cloud',
        colormap: str = 'viridis',
        max_words: int = 100
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Generate word cloud from texts.
        
        Parameters
        ----------
        texts : list or pd.Series
            Collection of texts
        title : str
            Plot title
        colormap : str
            Matplotlib colormap name
        max_words : int
            Maximum words to display
            
        Returns
        -------
        tuple
            (Figure, Axes) objects
        """
        if not HAS_WORDCLOUD:
            raise ImportError("wordcloud package required. Install with: pip install wordcloud")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Combine texts
        if isinstance(texts, pd.Series):
            texts = texts.dropna().tolist()
        combined_text = ' '.join(str(t) for t in texts)
        
        # Generate word cloud
        wc = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=max_words,
            colormap=colormap,
            collocations=False,
            random_state=42
        ).generate(combined_text)
        
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        return fig, ax
    
    def plot_sentiment_wordclouds(
        self,
        df: pd.DataFrame,
        text_col: str = 'processed_text',
        sentiment_col: str = 'predicted_sentiment'
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Generate side-by-side word clouds by sentiment.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with text and sentiment data
        text_col : str
            Column containing processed text
        sentiment_col : str
            Column containing sentiment labels
            
        Returns
        -------
        tuple
            (Figure, list of Axes) objects
        """
        if not HAS_WORDCLOUD:
            raise ImportError("wordcloud package required")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        sentiments = ['positive', 'negative', 'neutral']
        colormaps = ['Greens', 'Reds', 'Greys']
        
        for ax, sentiment, cmap in zip(axes, sentiments, colormaps):
            texts = df[df[sentiment_col] == sentiment][text_col]
            
            if len(texts) > 0:
                combined = ' '.join(texts.dropna().astype(str))
                
                wc = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    max_words=100,
                    colormap=cmap,
                    collocations=False,
                    random_state=42
                ).generate(combined)
                
                ax.imshow(wc, interpolation='bilinear')
                ax.set_title(f'{sentiment.capitalize()} Posts', fontweight='bold')
            ax.axis('off')
        
        plt.suptitle('Word Clouds by Sentiment', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return fig, axes
    
    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        title: str = 'Feature Correlation Matrix'
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create correlation heatmap.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with numerical columns
        columns : list, optional
            Columns to include. If None, uses all numerical columns.
        title : str
            Plot title
            
        Returns
        -------
        tuple
            (Figure, Axes) objects
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Select numerical columns
        if columns is None:
            numerical_df = df.select_dtypes(include=[np.number])
        else:
            numerical_df = df[columns]
        
        # Calculate correlation
        corr = numerical_df.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Heatmap
        if HAS_SEABORN:
            sns.heatmap(
                corr, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, square=True,
                linewidths=0.5, cbar_kws={'shrink': 0.8}, ax=ax
            )
        else:
            im = ax.imshow(np.ma.masked_array(corr, mask), cmap='RdBu_r', 
                          vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha='right')
            ax.set_yticklabels(corr.columns)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        return fig, ax
    
    def plot_hashtag_analysis(
        self,
        df: pd.DataFrame,
        hashtag_col: str = 'extracted_hashtags',
        score_col: str = 'compound_score',
        top_n: int = 20
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Create hashtag frequency and sentiment analysis.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with hashtag data
        hashtag_col : str
            Column containing hashtag lists
        score_col : str
            Column containing sentiment scores
        top_n : int
            Number of top hashtags to show
            
        Returns
        -------
        tuple
            (Figure, list of Axes) objects
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        
        # Flatten hashtags
        all_hashtags = []
        for tags in df[hashtag_col].dropna():
            if isinstance(tags, list):
                all_hashtags.extend(tags)
            elif isinstance(tags, str):
                all_hashtags.extend(tags.split(','))
        
        # Count hashtags
        hashtag_counts = Counter(all_hashtags)
        top_hashtags = hashtag_counts.most_common(top_n)
        
        # Calculate sentiment for each hashtag
        hashtag_sentiment = []
        for tag, count in top_hashtags:
            mask = df[hashtag_col].apply(
                lambda x: tag in x if isinstance(x, list) else tag in str(x)
            )
            mean_score = df.loc[mask, score_col].mean()
            hashtag_sentiment.append({
                'hashtag': f'#{tag}',
                'count': count,
                'sentiment': mean_score
            })
        
        hashtag_df = pd.DataFrame(hashtag_sentiment)
        
        # Plot frequency
        ax1 = axes[0]
        ax1.barh(hashtag_df['hashtag'][::-1], hashtag_df['count'][::-1],
                color='#3498db', alpha=0.8)
        ax1.set_xlabel('Number of Posts')
        ax1.set_title('Top Hashtags by Frequency', fontweight='bold')
        
        # Plot sentiment
        ax2 = axes[1]
        colors = [self.SENTIMENT_COLORS['positive'] if x > 0 
                  else self.SENTIMENT_COLORS['negative'] 
                  for x in hashtag_df['sentiment'][::-1]]
        ax2.barh(hashtag_df['hashtag'][::-1], hashtag_df['sentiment'][::-1],
                color=colors, alpha=0.8)
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Mean Compound Score')
        ax2.set_title('Top Hashtags by Sentiment', fontweight='bold')
        
        plt.tight_layout()
        return fig, axes
    
    def create_dashboard(
        self,
        df: pd.DataFrame,
        output_file: str = 'dashboard_overview.png'
    ) -> plt.Figure:
        """
        Create comprehensive dashboard with multiple visualizations.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with all required columns
        output_file : str
            Output filename
            
        Returns
        -------
        plt.Figure
            Dashboard figure
        """
        fig = plt.figure(figsize=(20, 16))
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Sentiment pie chart
        ax1 = fig.add_subplot(gs[0, 0])
        counts = df['predicted_sentiment'].value_counts()
        colors = [self.SENTIMENT_COLORS.get(s, '#ccc') for s in counts.index]
        ax1.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
               colors=colors, shadow=True)
        ax1.set_title('Sentiment Distribution', fontweight='bold')
        
        # 2. Compound score histogram
        ax2 = fig.add_subplot(gs[0, 1:])
        ax2.hist(df['compound_score'].dropna(), bins=50, color='#3498db', 
                alpha=0.7, edgecolor='black')
        ax2.axvline(x=df['compound_score'].mean(), color='#e74c3c', 
                   linestyle='--', linewidth=2)
        ax2.set_xlabel('Compound Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Score Distribution', fontweight='bold')
        
        # 3. Timeline (if timestamp exists)
        if 'timestamp' in df.columns:
            ax3 = fig.add_subplot(gs[1, :])
            daily = df.groupby(pd.to_datetime(df['timestamp']).dt.date)['compound_score'].mean()
            ax3.plot(daily.index, daily.values, color='#3498db', alpha=0.5)
            ax3.plot(daily.index, daily.rolling(7).mean(), color='#e74c3c', linewidth=2)
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Sentiment Score')
            ax3.set_title('Sentiment Timeline', fontweight='bold')
        
        # 4. Language comparison
        if 'language' in df.columns:
            ax4 = fig.add_subplot(gs[2, 0])
            lang_sent = df.groupby('language')['compound_score'].mean().sort_values()
            colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in lang_sent.values]
            ax4.barh(lang_sent.index, lang_sent.values, color=colors)
            ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax4.set_title('Sentiment by Language', fontweight='bold')
        
        # 5. Summary statistics
        ax5 = fig.add_subplot(gs[2, 1:])
        ax5.axis('off')
        
        stats_text = f"""
        ANALYSIS SUMMARY
        
        Total Posts: {len(df):,}
        
        Sentiment Distribution:
        • Positive: {(df['predicted_sentiment']=='positive').sum():,} ({(df['predicted_sentiment']=='positive').mean()*100:.1f}%)
        • Negative: {(df['predicted_sentiment']=='negative').sum():,} ({(df['predicted_sentiment']=='negative').mean()*100:.1f}%)
        • Neutral: {(df['predicted_sentiment']=='neutral').sum():,} ({(df['predicted_sentiment']=='neutral').mean()*100:.1f}%)
        
        Sentiment Scores:
        • Mean: {df['compound_score'].mean():.4f}
        • Median: {df['compound_score'].median():.4f}
        • Std Dev: {df['compound_score'].std():.4f}
        """
        
        ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('COVID-19 Instagram Sentiment Analysis Dashboard', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Save
        self.save_figure(output_file, fig)
        
        return fig


if __name__ == '__main__':
    # Example usage
    print("Visualization module loaded successfully!")
    print(f"Seaborn available: {HAS_SEABORN}")
    print(f"WordCloud available: {HAS_WORDCLOUD}")
