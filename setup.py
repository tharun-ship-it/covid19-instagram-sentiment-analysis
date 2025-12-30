"""
Setup configuration for COVID-19 Instagram Sentiment Analysis package.

Install in development mode:
    pip install -e .

Install with extras:
    pip install -e ".[dev]"
    pip install -e ".[deep_learning]"

Author: Tharun Ponnam
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / 'README.md'
if readme_path.exists():
    long_description = readme_path.read_text(encoding='utf-8')
else:
    long_description = 'COVID-19 Instagram Sentiment Analysis'

# Core dependencies
install_requires = [
    'numpy>=1.21.0',
    'pandas>=1.3.0',
    'matplotlib>=3.4.0',
    'seaborn>=0.11.0',
    'nltk>=3.6.0',
    'pyyaml>=5.4.0',
    'tqdm>=4.62.0',
]

# Optional dependencies
extras_require = {
    'dev': [
        'pytest>=6.2.0',
        'pytest-cov>=2.12.0',
        'black>=21.0',
        'flake8>=3.9.0',
        'isort>=5.9.0',
        'jupyter>=1.0.0',
    ],
    'multilingual': [
        'langdetect>=1.0.9',
    ],
    'visualization': [
        'wordcloud>=1.8.0',
        'plotly>=5.3.0',
    ],
    'deep_learning': [
        'tensorflow>=2.6.0',
        'transformers>=4.10.0',
    ],
}

# All extras
extras_require['all'] = list(set(
    pkg for extras in extras_require.values() for pkg in extras
))

setup(
    name='covid19-instagram-sentiment',
    version='1.0.0',
    author='Tharun Ponnam',
    author_email='tharunponnam007@gmail.com',
    description='Multilingual sentiment analysis of COVID-19 discourse on Instagram',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/tharun-ship-it/covid19-instagram-sentiment-analysis',
    project_urls={
        'Bug Tracker': 'https://github.com/tharun-ship-it/covid19-instagram-sentiment-analysis/issues',
        'Documentation': 'https://github.com/tharun-ship-it/covid19-instagram-sentiment-analysis#readme',
        'Source': 'https://github.com/tharun-ship-it/covid19-instagram-sentiment-analysis',
    },
    packages=find_packages(exclude=['tests', 'tests.*', 'notebooks']),
    python_requires='>=3.8',
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing :: Linguistic',
    ],
    keywords=[
        'sentiment-analysis',
        'nlp',
        'covid-19',
        'instagram',
        'social-media',
        'multilingual',
        'text-analysis',
        'data-science',
    ],
    entry_points={
        'console_scripts': [
            'covid-sentiment=src.sentiment_analyzer:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
