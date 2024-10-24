import pandas as pd
import json
from prophet import Prophet
import matplotlib.pyplot as plt
import time

# Step 1: Load demand forecast data from CSV
demand_forecast = pd.read_csv('/home/heathknowles/Documents/AINN Project/updated_file.csv')  # Replace with your CSV path
demand_forecast['ds'] = pd.to_datetime(demand_forecast['ds'])

# Step 2: Load sentiment data from JSON file
with open('sentiment.json', 'r') as f:
    sentiment_data = json.load(f)

# Step 3: Load prices from JSON file
with open('prices.json', 'r') as f:
    prices = json.load(f)

# Clean and convert prices to integers (remove commas and convert to integer)
prices = [int(price.replace(",", "")) for price in prices]

# Step 4: Load optimized prices from CSV (if needed)
optimized_prices = pd.read_csv('/home/heathknowles/Documents/AINN Project/optimized_prices.csv')  # Replace with your CSV path

# Step 5: Forecast demand using Prophet
# Prepare data for Prophet
df = pd.DataFrame({
    'ds': demand_forecast['ds'],
    'y': demand_forecast['yhat']
})

# Initialize the Prophet model and fit it on the historical demand data
model = Prophet()
model.fit(df)

# Make future dataframe for forecast (let's forecast for the next 12 months, so adjust as needed)
future = model.make_future_dataframe(periods=12, freq='M')  # Forecasting 12 months ahead

# Predict the demand for future months
forecast = model.predict(future)

# Step 6: Get the specific month for which to forecast the price
specific_month = '2024-10'  # Set the specific month you want (e.g., October 2024)
print(f"Forecasting for specific month: {specific_month}")

# Step 7: Filter the forecast for the specific month
forecasted_demand_month = forecast[forecast['ds'].dt.strftime('%Y-%m') == specific_month]

if forecasted_demand_month.empty:
    print(f"No forecast available for {specific_month}. Please check the forecast range.")
else:
    # Step 8: Aggregate demand for the month
    aggregated_demand = forecasted_demand_month['yhat'].mean()

    # Step 9: Calculate a single adjusted price for the specific month based on forecasted demand and sentiment
    def calculate_single_adjusted_price(prices, forecasted_demand, sentiment_data, demand_threshold=50, price_increase=1.1, price_decrease=0.9):
        # Compute a sentiment score as a weighted sum from the sentiment data list
        sentiment_score = (
            sentiment_data[0]['score'] * 1.2 +  # Positive sentiment
            sentiment_data[1]['score'] * 0.8 +  # Negative sentiment
            sentiment_data[2]['score'] * 1.0    # Neutral sentiment
        )
        
        # Choose the base price (let's use the average of the given prices as the base price)
        base_price = sum(prices) / len(prices)
        
        # Adjust price based on aggregated demand for the specific month
        if aggregated_demand > demand_threshold:
            # If demand is higher than the threshold, increase the price
            adjusted_price = base_price * price_increase * sentiment_score
        else:
            # If demand is lower than the threshold, decrease the price
            adjusted_price = base_price * price_decrease * sentiment_score
        
        return adjusted_price

    # Apply price adjustment based on aggregated demand and sentiment for the specific month
    single_adjusted_price = calculate_single_adjusted_price(prices, forecasted_demand_month, sentiment_data)

    # Print the result (the single adjusted price for the specific month)
    print(f"Adjusted price for {specific_month}: {single_adjusted_price:.2f}")

# Step 10: Plot the demand forecast and save as PNG
    plt.figure(figsize=(10, 6))
    plt.plot(forecast['ds'], forecast['yhat'], label="Forecasted Demand", color='b')
    plt.axvline(x=pd.to_datetime(specific_month), color='r', linestyle='--', label=f'Specific Month: {specific_month}')
    plt.title("Demand Forecast Over Time")
    plt.xlabel("Date")
    plt.ylabel("Demand (yhat)")
    plt.legend()
    plt.grid(True)
    
    # Save the plot to a PNG file
    plt.savefig(f'demand_forecast_{specific_month}.png')
    print(f"Demand forecast plot saved as 'demand_forecast_{specific_month}.png'")

    # Step 11: Plot Adjusted Prices for Each Month and save as PNG
    months = [time.strftime('%Y-%m', time.strptime(forecast['ds'].iloc[i].strftime('%Y-%m'), '%Y-%m')) for i in range(len(forecast['ds']))]
    
    adjusted_prices = []
    for month in months:
        # Recalculate adjusted price for each month
        monthly_demand = forecast[forecast['ds'].dt.strftime('%Y-%m') == month]['yhat'].mean()
        price = calculate_single_adjusted_price(prices, forecast[forecast['ds'].dt.strftime('%Y-%m') == month], sentiment_data)
        adjusted_prices.append(price)

    plt.figure(figsize=(10, 6))
    plt.plot(months, adjusted_prices, label="Adjusted Prices", color='g')
    plt.axvline(x=specific_month, color='r', linestyle='--', label=f"Adjusted Price for {specific_month}")
    plt.title("Adjusted Prices Over Time")
    plt.xlabel("Month")
    plt.ylabel("Adjusted Price")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot to a PNG file
    plt.savefig(f'adjusted_prices_{specific_month}.png')
    print(f"Adjusted prices plot saved as 'adjusted_prices_{specific_month}.png'")
