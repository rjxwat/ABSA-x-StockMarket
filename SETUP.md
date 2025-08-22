# Quick Setup Guide

## Installation

1. **Install Python packages**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download NLP models**:
   ```bash
   python -m spacy download en_core_web_sm
   python -m nltk.downloader stopwords punkt wordnet
   ```

3. **For web scraping (optional)**:
   - Download [ChromeDriver](https://chromedriver.chromium.org/)
   - Add to your system PATH

## Running the Project

1. **News Collection**: `python news_extraction/webscraping.py`
2. **Sentiment Analysis**: Run notebooks in `aspects/` folder
3. **Text Encoding**: Run notebooks in `Syntactic_encoder/` and `numerical_encoder/`
4. **Stock Prediction**: `python stockmovementpredictor/stockmovementpredictor.py`

## Notes

- Some notebooks work better on Kaggle (GPU required)
- Check individual folder README files for specific instructions
- Make sure you have sufficient memory for large datasets
