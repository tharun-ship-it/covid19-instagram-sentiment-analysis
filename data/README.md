# Dataset Documentation

## COVID-19 Instagram Discourse Dataset

### Overview

This directory contains the dataset used for multilingual sentiment analysis of COVID-19 discourse on Instagram. The dataset spans five years of pandemic-related posts and is published as a peer-reviewed research resource.

### Source & Attribution

| Attribute | Details |
|-----------|---------|
| **Creator** | Nirmalya Thakur, Ph.D. |
| **Official Provider** | [IEEE DataPort](https://ieee-dataport.org/documents/five-years-covid-19-discourse-instagram-labeled-instagram-dataset-over-half-million-posts) |
| **DOI** | 10.21227/d46p-v480 |
| **Open Access Mirror** | [Zenodo Record 13896353](https://zenodo.org/records/13896353) |
| **Associated Paper** | [IEEE MLNLP 2024](https://ieeexplore.ieee.org/document/10800025) |

### Download Instructions

#### Option 1: Manual Download (Recommended)

1. Visit [Zenodo](https://zenodo.org/records/13896353) (free, no subscription required)
2. Download the dataset file(s)
3. Place them in this `data/` directory

#### Option 2: IEEE DataPort

1. Visit [IEEE DataPort](https://ieee-dataport.org/documents/five-years-covid-19-discourse-instagram-labeled-instagram-dataset-over-half-million-posts)
2. Requires IEEE subscription OR use Zenodo link provided on the page

#### Option 3: Automated Download

```bash
# Using the provided data loader
python src/data_loader.py --download

# Or using wget
wget https://zenodo.org/records/13896353/files/instagram_covid19_dataset.csv -P data/
```

### Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `post_id` | string | Unique Instagram post identifier |
| `text` | string | Post caption/content (UTF-8 encoded) |
| `timestamp` | datetime | Publication time (UTC) |
| `language` | string | ISO 639-1 language code |
| `sentiment_label` | string | Annotated sentiment class |
| `engagement_score` | float | Normalized engagement (0-1) |
| `hashtags` | list | Extracted hashtags |
| `emoji_count` | int | Number of emojis in post |

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Records | 500,153 |
| Languages | 161 |
| Time Span | Jan 2020 – Dec 2024 |
| Avg. Post Length | 156 characters |

### Language Distribution

The dataset contains posts in **161 languages**, with the top languages being:

| Language | Code | Approximate % |
|----------|------|---------------|
| English | en | ~68% |
| Spanish | es | ~12% |
| Portuguese | pt | ~7% |
| French | fr | ~4% |
| German | de | ~2% |
| Italian | it | ~2% |
| Others (155 languages) | — | ~5% |

### Sentiment Distribution

| Label | Count | Percentage |
|-------|-------|------------|
| Positive | 227,546 | 42.3% |
| Negative | 171,103 | 31.8% |
| Neutral | 139,096 | 25.9% |

### Data Quality Notes

1. **Deduplication**: Near-duplicate posts (Levenshtein distance < 0.15) have been removed
2. **Encoding**: All text is UTF-8 normalized
3. **Missing Values**: Posts with missing text content have been excluded
4. **Timestamps**: All times are normalized to UTC

### Ethical Considerations

- This dataset is intended for research purposes only
- User identifiers have been anonymized
- The dataset complies with Instagram's Terms of Service for research
- Please cite the original data sources when using this dataset

### Citation

When using this dataset, please cite the original paper:

```bibtex
@inproceedings{thakur2024covid19instagram,
  author    = {Thakur, Nirmalya},
  title     = {Five Years of COVID-19 Discourse on Instagram: A Labeled 
               Instagram Dataset of Over Half a Million Posts for 
               Multilingual Sentiment Analysis},
  booktitle = {Proceedings of the 7th International Conference on Machine 
               Learning and Natural Language Processing (IEEE MLNLP 2024)},
  year      = {2024},
  address   = {Chengdu, China},
  month     = {October},
  doi       = {10.21227/d46p-v480},
  url       = {https://ieeexplore.ieee.org/document/10800025}
}
```

### License

Please refer to the original dataset license on Zenodo for usage terms.

### File Structure

After downloading, your `data/` directory should contain:

```
data/
├── README.md                          # This file
├── instagram_covid19_dataset.csv      # Main dataset (download required)
└── processed/                         # Generated after analysis
    ├── sentiment_results.csv
    └── aggregated_statistics.json
```

### Troubleshooting

**Issue**: Download fails with authentication error
- Solution: Register for a free Zenodo account if required

**Issue**: File encoding errors during loading
- Solution: The DataLoader handles multiple encodings automatically

**Issue**: Dataset too large for memory
- Solution: Use sampling: `loader.load_dataset(sample_size=100000)`

### Contact

For dataset-related questions, please refer to the original authors on IEEE DataPort or Zenodo.

For questions about this analysis project, contact:
- **Author**: Tharun Ponnam
- **Email**: tharunponnam007@gmail.com
