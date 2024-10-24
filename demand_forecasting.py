import os
import pandas as pd
from pytrends.request import TrendReq
import time
from datetime import datetime, timedelta
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
from prophet import Prophet
import json
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# Create the trends_data folder if it doesn't exist
if not os.path.exists("trends_data"):
    os.makedirs("trends_data")

# Function to scrape product release date from a Wikipedia page
def scrape_wikipedia_release_date(product_name):
    search_url = f"https://en.wikipedia.org/wiki/{product_name.replace(' ', '_')}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    response = requests.get(search_url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        release_date = None
        info_box = soup.find("table", class_="infobox")
        
        if info_box:
            for row in info_box.find_all("tr"):
                if 'release date' in row.text.lower():
                    release_date = row.find("td").text.strip()
                    break

        if release_date:
            print(f"Release date found: {release_date}")
        else:
            print("Couldn't find release date on the Wikipedia page.")
        return release_date
    else:
        print(f"Failed to retrieve Wikipedia page, status code: {response.status_code}")
        return None

# Function to clean and convert the Wikipedia date to datetime format (dd-mm-yy)
def parse_wikipedia_release_date(date_str, keyword):
    release_dates = date_str.split("CHN:")
    for release_date in release_dates:
        if keyword.lower() in release_date.lower():
            try:
                return datetime.strptime(release_date.split(":")[-1].strip(), "%B %d, %Y")
            except ValueError:
                try:
                    return datetime.strptime(release_date.split(":")[-1].strip(), "%Y-%m-%d")
                except ValueError:
                    continue
    return None

# Function to adjust release date to 2017 if it's earlier
def adjust_release_date(release_date):
    if release_date.year < 2017:
        print(f"Product release date {release_date.strftime('%Y-%m-%d')} is before 2017. Adjusting to 2017-01-01.")
        release_date = datetime(2017, 1, 1)
    return release_date

# Function to fetch Google Trends data for a single month
def get_google_trends_data_for_month(pytrends, keywords, start_date, pbar):
    timeframe = f"{start_date.strftime('%Y-%m-%d')} {(start_date + timedelta(days=32)).replace(day=1) - timedelta(days=1):%Y-%m-%d}"
    
    try:
        pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo='')
        interest_over_time_df = pytrends.interest_over_time()
        
        pbar.update(1)
        
        return interest_over_time_df
    except Exception as e:
        print(f"Error fetching data for {timeframe}: {str(e)}")
        return pd.DataFrame()

# Function to fetch Google Trends data for all months
def get_google_trends_data(keywords, start_date, end_date):
    pytrends = TrendReq(hl='en-US', tz=360)
    
    total_months = (end_date.year - start_date.year) * 12 + end_date.month - start_date.month + 1
    pbar = tqdm(total=total_months, desc="Fetching monthly data")
    
    all_data = []
    current_date = start_date
    
    while current_date <= end_date:
        monthly_data = get_google_trends_data_for_month(pytrends, keywords, current_date, pbar)
        if not monthly_data.empty:
            all_data.append(monthly_data)
        
        current_date += timedelta(days=32)
        current_date = current_date.replace(day=1)
    
    pbar.close()
    
    if all_data:
        return pd.concat(all_data)
    else:
        return pd.DataFrame()

# Function to forecast demand using Prophet model
def forecast_demand(df, keyword):
    # Convert the 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Rename the columns to match Prophet's expected format
    df.rename(columns={'date': 'ds', keyword: 'y'}, inplace=True)

    # Ensure the data is sorted by date
    df = df.sort_values(by='ds')

    # Initialize the Prophet model
    model = Prophet()

    # Fit the model on the historical data
    model.fit(df)

    # Create a future dataframe to forecast for the next 12 months (adjust periods if needed)
    future = model.make_future_dataframe(periods=12, freq='M')

    # Forecast future demand
    forecast = model.predict(future)

    # Extract the forecasted values (yhat) and the corresponding dates (ds)
    forecast_values = forecast[['ds', 'yhat']]

    # Save forecast to CSV
    forecast_filename = f"trends_data/forecast_{keyword}.csv"
    forecast_values.to_csv(forecast_filename, index=False)
    print(f"Forecasted data saved to {forecast_filename}")

    # Convert forecast to JSON format
    forecast_json = forecast_values.to_json(orient="records", date_format="iso")
    print(forecast_json)

    return forecast_values

# Function to calculate and display model performance metrics
def evaluate_forecast(forecast, actual):
    # Ensure the data is aligned by date
    actual = actual.set_index('ds')
    forecast = forecast.set_index('ds')

    # Merge the actual and forecasted values on the 'ds' column (date)
    merged = actual.join(forecast[['yhat']], on='ds', how='inner')
    
    # Calculate evaluation metrics
    y_true = merged['y'].values
    y_pred = merged['yhat'].values
    
    # R-squared score
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error (MAPE)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Print metrics
    print(f"R-squared: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"RMSE: {rmse:.2f}")
    
    return r2, mape

# Function to forecast and evaluate Prophet model performance
def forecast_and_evaluate(df, keyword):
    # Convert the 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Rename the columns to match Prophet's expected format
    df.rename(columns={'date': 'ds', keyword: 'y'}, inplace=True)

    # Ensure the data is sorted by date
    df = df.sort_values(by='ds')

    # Train-test split: Use 80% of the data for training and the remaining 20% for testing
    train_size = int(0.8 * len(df))
    train_df = df[:train_size]
    test_df = df[train_size:]

    # Initialize and train the Prophet model
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)

    # Fit the model on the training data
    model.fit(train_df)

    # Create a future dataframe to forecast for the test period
    future = model.make_future_dataframe(periods=len(test_df), freq='M')

    # Forecast future demand
    forecast = model.predict(future)

    # Evaluate the forecast on the test set
    r2, mape = evaluate_forecast(forecast, test_df)
    
    # If performance is below threshold, adjust parameters
    if mape > 4:  # A MAPE of 4% or better would be close to 96% accuracy
        print("MAPE is greater than 4%. Tuning the model...")
        # You can adjust the changepoint prior scale or seasonality scale for better performance
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                        changepoint_prior_scale=0.05)  # Smaller changepoint scale for smoother trend
        model.fit(train_df)
        forecast = model.predict(future)
        r2, mape = evaluate_forecast(forecast, test_df)

    # Save forecast to CSV
    forecast_filename = f"trends_data/forecast_{keyword}.csv"
    forecast[['ds', 'yhat']].to_csv(forecast_filename, index=False)
    print(f"Forecasted data saved to {forecast_filename}")

    # Return the final forecast
    return forecast[['ds', 'yhat']].to_json(orient="records", date_format="iso")


# Function to generate keywords based on the product name
def generate_keywords(product_name):
    # Basic keywords based on product name
    keywords = [product_name]
    
    # Add common variations of the product name
    keywords.append(f"{product_name} console")
    keywords.append(f"{product_name} games")
    
    # Example of adding a plural form
    if not product_name.endswith('s'):
        keywords.append(f"{product_name}s")
    
    return keywords

# Main function to scrape release date from Wikipedia, fetch Google Trends data, and forecast demand
def main():
    print("Google Trends Data Downloader and Forecast Processor")
    
    # Specify the product name directly
    product_name = "Nintendo Switch"  # Replace with your desired product name
    
    # Generate keywords based on product name
    keywords = generate_keywords(product_name)
    print(f"Generated keywords for {product_name}: {', '.join(keywords)}")
    
    # Scrape Wikipedia release date for the product
    release_date_str = scrape_wikipedia_release_date(product_name)
    
    if not release_date_str:
        print(f"No valid release date found for {product_name}. Exiting.")
        return
    
    print(f"Looking for release date for {product_name}...")
    release_date = parse_wikipedia_release_date(release_date_str, product_name)
    
    if release_date:
        print(f"Release date for {product_name}: {release_date.strftime('%Y-%m-%d')}")
    else:
        print(f"No specific release date found for {product_name}. Using default release date of 2017.")
        release_date = datetime(2017, 1, 1)
    
    release_date = adjust_release_date(release_date)
    start_date = release_date
    end_date = datetime.now()
    
    print(f"Downloading monthly data for keyword: {product_name}")
    try:
        # Fetch Google Trends data for the product name
        trends_data = get_google_trends_data(keywords, start_date, end_date)
        
        if trends_data.empty:
            print(f"No data found for {product_name}.")
            return

        # Preprocess the dataset
        trends_data.reset_index(inplace=True)
        trends_data = trends_data[['date', product_name]].dropna()

        # Forecast demand using Prophet and evaluate performance
        forecast_json = forecast_and_evaluate(trends_data, product_name)
        print(f"Forecast data in JSON format: {forecast_json}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
