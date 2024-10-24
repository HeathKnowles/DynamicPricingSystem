import os
import time
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.optimize import linprog
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options

# Define your single product name here
product_name = "Nintendo Switch oled white"

def load_prices_from_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    if isinstance(data, list):
        return [int(p.replace(',', '')) for p in data]
    elif isinstance(data, dict) and "prices" in data:
        return [int(p.replace(',', '')) for p in data["prices"]]
    else:
        raise ValueError("Invalid JSON structure. Expected an array or a dictionary with 'prices' key.")

def dynamic_pricing(prices):
    max_price = max(prices)
    min_price = min(prices)
    adjusted_prices = [(p - min_price) / (max_price - min_price) * (max_price - min_price) + min_price for p in prices]
    return adjusted_prices

def price_elasticity(prices, elasticity_factor=0.05):
    adjusted_prices = [p * (1 - elasticity_factor) for p in prices]
    return adjusted_prices

# Linear Programming (Simple maximization of profit)
def linear_programming(prices):
    c = [-p for p in prices]  
    A = [[1] * len(prices)]
    b = [len(prices) * 1000]  
    result = linprog(c, A_ub=A, b_ub=b, method='highs')
    return result.x

def machine_learning_pricing(prices):
    X = np.array(range(len(prices))).reshape(-1, 1)
    y = np.array(prices)
    model = LinearRegression()
    model.fit(X, y)
    future_price = model.predict([[len(prices)]])
    return future_price[0]

# A/B Testing (Randomly split into two groups with different prices)
def ab_testing(prices):
    group_a = prices[:len(prices)//2]
    group_b = prices[len(prices)//2:]
    revenue_a = sum(group_a)
    revenue_b = sum(group_b)
    return group_a, group_b, revenue_a, revenue_b

# Game Theory (Simple competitive pricing model)
def game_theory(prices):
    competitor_prices = [p * 0.9 for p in prices]  
    best_response = [max(p, c) for p, c in zip(prices, competitor_prices)]
    return best_response

def behavioral_pricing(prices, discount=0.1):
    discounted_prices = [p * (1 - discount) for p in prices]
    return discounted_prices

def visit_product_page(driver, url):
    driver.get(url)
    time.sleep(3)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    table_body = soup.find("tbody", id="sh-osd__online-sellers-cont")
    prices_in_inr = []
    delivery_costs = []
    if table_body:
        for row in table_body.find_all("tr"):
            row_text = row.get_text().lower()
            if 'price' in row_text and '₹' in row_text:
                price_parts = [part.strip() for part in row_text.split() if 'price' in part and '₹' in part]
                filtered_prices = [price for price in price_parts if '₹0.00' not in price]
                prices_in_inr.extend(filtered_prices)
                if 'delivery' in row_text:
                    delivery_costs.extend([part.strip() for part in row_text.split() if '₹' in part and 'delivery' in part])
    cleaned_prices = clean_and_sum_prices(prices_in_inr, delivery_costs)
    save_to_json(cleaned_prices)

def clean_and_sum_prices(prices, delivery_costs):
    cleaned_prices = []
    for price in prices:
        cleaned_price = price.replace('visit', '').replace('delivery', '').replace('total', '')
        cleaned_price = cleaned_price.replace("\u20b9", "₹").strip()
        cleaned_price = ''.join(c for c in cleaned_price if c.isdigit() or c == '.' or c == ',')
        if '.' in cleaned_price:
            price_parts = cleaned_price.split('.')
            if len(price_parts) > 2:
                cleaned_price = price_parts[0] + '.' + price_parts[1]
        if cleaned_price and cleaned_price != '0.00':
            cleaned_prices.append(cleaned_price)
    
    cleaned_prices_no_decimal = [price.split('.')[0] if '.' in price else price for price in cleaned_prices]
    return cleaned_prices_no_decimal

def save_to_json(data):
    with open("prices.json", "a") as json_file:
        json.dump(data, json_file, indent=4)
        json_file.write("\n")
    print("Prices and totals saved to prices.json")

def google_search(queries):
    options = Options()
    options.headless = True
    options.binary_location = "/usr/bin/firefox"
    driver_service = Service(executable_path="/home/heathknowles/Documents/AINN Project/geckodriver")
    driver = webdriver.Firefox(service=driver_service, options=options)

    for query in queries:
        search_url = f"https://www.google.com/search?q={query}&hl=en&tbm=shop&psb=1"
        driver.get(search_url)
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        search_results = soup.find_all("a", class_="_-pS")
        for result in search_results:
            href = result.get("href")
            if href:
                full_url = f"https://www.google.com{href}" if href.startswith('/') else href
                print(f"Visiting: {full_url}")
                visit_product_page(driver, full_url)
    driver.quit()

# Main function
if __name__ == "__main__":
    # Single product processing
    queries = [product_name]
    
    google_search(queries)
    
    # Load scraped prices
    prices = load_prices_from_json('prices.json')

    # Apply all pricing algorithms
    dynamic_prices = dynamic_pricing(prices)
    elasticity_prices = price_elasticity(prices)
    lp_prices = linear_programming(prices)
    ml_price = machine_learning_pricing(prices)
    group_a, group_b, revenue_a, revenue_b = ab_testing(prices)
    game_theory_prices = game_theory(prices)
    behavioral_prices = behavioral_pricing(prices)

    # Print results
    print("Original Prices:", prices)
    print("Dynamic Pricing Adjusted Prices:", dynamic_prices)
    print("Elasticity Pricing Adjusted Prices:", elasticity_prices)
    print("Linear Programming Prices:", lp_prices)
    print("ML Predicted Future Price:", ml_price)
    print("A/B Testing Group A Prices:", group_a)
    print("A/B Testing Group B Prices:", group_b)
    print("Revenue from Group A:", revenue_a)
    print("Revenue from Group B:", revenue_b)
    print("Game Theory Best Response Prices:", game_theory_prices)
    print("Behavioral Pricing Discounted Prices:", behavioral_prices)

    # Save results to CSV
    results_df = pd.DataFrame({
        "Original Prices": prices,
        "Dynamic Pricing": dynamic_prices,
        "Elasticity Pricing": elasticity_prices,
        "Linear Programming": lp_prices[:len(prices)],
        "Behavioral Pricing": behavioral_prices
    })

    results_df.to_csv("optimized_prices.csv", index=False)