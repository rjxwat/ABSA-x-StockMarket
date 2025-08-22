# Financial News Data Collection

This module collects financial news data from various online sources for analysis.

## Contents

### 1. **webscraping.py**
- Scrapes financial news from Economic Times website
- **Libraries used**: Beautiful Soup (HTML parsing), Selenium (dynamic content), Pandas (data handling)
- **Features**: Date-range filtering, company-specific news collection, automated pagination

### 2. **news_api.ipynb**
- Uses news APIs for additional data collection
- **Features**: API integration, real-time data, company-specific queries

## Usage

### Web Scraping
```python
cd news_extraction
python webscraping.py
```

### API Collection
```bash
jupyter notebook news_api.ipynb
```

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. For Selenium: Download ChromeDriver and add to PATH
3. Configure API keys in news_api.ipynb (if using)

## Output

The scripts generate CSV files with:
- Date, Headlines, Source, URL, Company
- Cleaned and structured news data for analysis

## Note

- Respect website terms of service
- Use appropriate delays between requests
- Some functionality may require API keys

