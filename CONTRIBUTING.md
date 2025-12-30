# Contributing to COVID-19 Instagram Sentiment Analysis

First off, thank you for considering contributing to this project! 🎉

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Style Guidelines](#style-guidelines)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project and everyone participating in it is governed by our commitment to maintaining a welcoming and inclusive environment. By participating, you are expected to uphold this standard. Please be respectful and constructive in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/covid19-instagram-sentiment-analysis.git
   cd covid19-instagram-sentiment-analysis
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/tharun-ship-it/covid19-instagram-sentiment-analysis.git
   ```

## Development Setup

### Prerequisites

- Python 3.8 or higher
- pip 21.0 or higher
- Git

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Download NLTK resources
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

### Verify Installation

```bash
# Run tests
pytest tests/ -v

# Run linting
flake8 src/
black --check src/
```

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:

- A clear, descriptive title
- Steps to reproduce the problem
- Expected vs actual behavior
- Your environment details (OS, Python version, etc.)
- Any relevant error messages or logs

### Suggesting Features

Feature suggestions are welcome! Please create an issue with:

- A clear description of the feature
- The problem it would solve
- Any alternative solutions you've considered
- If possible, a rough implementation idea

### Contributing Code

1. **Create a branch** for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes** following our style guidelines

3. **Write or update tests** for your changes

4. **Run the test suite** to ensure everything passes:
   ```bash
   pytest tests/ -v --cov=src
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   # or
   git commit -m "fix: fix your bug description"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request** on GitHub

## Style Guidelines

### Python Code Style

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 88 characters (Black default)
- **Imports**: Use `isort` for import ordering
- **Formatting**: Use `black` for code formatting

```bash
# Format code
black src/ tests/
isort src/ tests/

# Check style
flake8 src/ tests/
```

### Docstrings

Use NumPy style docstrings:

```python
def analyze_sentiment(text: str) -> Dict[str, float]:
    """
    Analyze the sentiment of a text.

    Parameters
    ----------
    text : str
        The input text to analyze.

    Returns
    -------
    dict
        Dictionary containing sentiment scores:
        - 'positive': float between 0 and 1
        - 'negative': float between 0 and 1
        - 'neutral': float between 0 and 1
        - 'compound': float between -1 and 1

    Examples
    --------
    >>> result = analyze_sentiment("I love this!")
    >>> print(result['compound'])
    0.6369
    """
    pass
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Test additions or modifications
- `chore:` Maintenance tasks

Examples:
```
feat: add multilingual sentiment support
fix: handle empty text input gracefully
docs: update README with new examples
test: add tests for emoji handling
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_analyzer.py -v

# Run specific test function
pytest tests/test_analyzer.py::TestSentimentAnalyzer::test_positive_sentiment -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use descriptive test names that explain what is being tested
- Include both positive and negative test cases

Example:

```python
def test_analyze_positive_text_returns_positive_sentiment():
    """Test that clearly positive text is classified correctly."""
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze("I love this! It's amazing!")
    
    assert result['label'] == 'positive'
    assert result['compound'] > 0.5
```

## Pull Request Process

1. **Update documentation** if you're adding or changing functionality

2. **Ensure all tests pass** and add new tests for your changes

3. **Update the CHANGELOG** if applicable

4. **Fill out the PR template** with:
   - Description of changes
   - Related issue numbers
   - Type of change (bug fix, feature, etc.)
   - Checklist completion

5. **Request a review** from maintainers

6. **Address review feedback** promptly

7. **Squash commits** if requested before merging

### PR Checklist

- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] Code follows style guidelines
- [ ] Commit messages follow conventions
- [ ] No merge conflicts with main branch

## Questions?

If you have questions, feel free to:

- Open an issue with the `question` label
- Reach out to the maintainer at tharunponnam007@gmail.com

Thank you for contributing! 🙏
