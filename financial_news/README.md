# Dataset Description for Financial News

## Overview

We have created datasets for 3 companies for now that is RELIANCE, APOLLO AND TATA. For now we have done aspect labelling for all 3 but syntactic encoding for only Reliance financial news. 

## Common Columns Across Datasets

The following columns are same for  the `reliance.csv`, `apollo.csv`, and `tata.csv` datasets:

1. **Date**: 
   - **Format**: `dd/mm/yy`
   - **Description**: The date on which the news article was published.

2. **news**: 
   - **Description**: Collection of financial news for taht day

3. **predicted Sentiment**: 
   - **Description**: The overall sentiment of the news article,either  positive, negative, or neutral.

4. **aspect_labelling**: 
   - **Description**: Contains a list of sets for each news article, detailing:
     - Aspect words: aspect word present in the sentence 
     - Categories: Categories associated with each aspect (e.g., Finance, Market).
     - Sentiment: The sentiment associated with each aspect (positive, negative, neutral).
     - Sentiment Score: A numerical value representing the sentiment strength for each aspect.

## Additional Columns in `reliance.csv`

In addition to the common columns, the `reliance.csv` dataset includes the following specific columns:

5. **syntactic_embeddings**: 
   - **Description**: This column contains syntactic embeddings generated for each news article. These embeddings capture the syntactic structure of the text.
6. **cleaned text**: 
   - **Description**: This column contains the preprocessed version of the news articles.
