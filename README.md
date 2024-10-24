# Dynamic Pricing System

This project implements a dynamic pricing system using Facebook Prophet for time series forecasting and Twitter RoBERTa for sentiment analysis. It adjusts prices dynamically based on future demand predictions and real-time customer sentiment analysis from Twitter data.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The dynamic pricing system leverages two key components:
1. **Prophet:** A forecasting tool to predict future demand based on historical data.
2. **Twitter RoBERTa:** A pre-trained transformer model for sentiment analysis to gauge real-time market sentiment from Twitter data.

Using these models, the system adjusts product prices to maximize revenue or competitiveness depending on predicted demand and social sentiment.

## Technologies Used
- Python 3.8+
- [Prophet](https://facebook.github.io/prophet/)
- [Hugging Face RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)
- pandas, numpy, scikit-learn
- Tweepy (for Twitter API)
- Matplotlib for visualization

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/HeathKnowles/DynamicPricingSystem.git
   cd DynamicPricingSystem
   ```
2. Run the data fetchers
   ```bash
   python commerce.py
   python demand_forecasting.py
   python sentiment_analysis.py
   ```

3. Run the preprocessor
   ```bash
   python preprocess.py
   ```

3. Run the main python file
   ```bash
   python main.py
   ```
4. This outputs a dynamic Price of the product in the json folder
   
    
