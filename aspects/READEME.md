# Sentiment Analysis and Aspect-Based Labeling

This folder contains tools for performing sentiment analysis and aspect-based labeling on financial news datasets.

## Contents

- **`sentiment-score.ipynb`**: Overall sentiment analysis for financial news articles using FinBERT model
- **`aspect-extraction.ipynb`**: Aspect-based sentiment analysis - extracts specific financial aspects and their sentiments

## Key Features

- **FinBERT Model**: Financial domain-specific BERT for accurate sentiment analysis
- **Aspect Categories**: 12 financial categories (stock_market, financial_performance, investor_sentiment, etc.)
- **Multi-level Analysis**: Both document-level and aspect-level sentiment scoring

## Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Download spaCy model: `python -m spacy download en_core_web_sm`
3. Run the notebooks in order:
   - First: `sentiment-score.ipynb` for overall sentiment
   - Second: `aspect-extraction.ipynb` for aspect-based analysis

## Note

These notebooks are optimized for **Kaggle execution** due to GPU requirements:
- [Sentiment Score Kaggle Notebook](https://www.kaggle.com/code/adi253/sentiment-score)
- [Aspect Extraction Kaggle Notebook](https://www.kaggle.com/code/adi253/aspect-extraction)



