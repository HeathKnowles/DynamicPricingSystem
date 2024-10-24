import os
import sys
import csv
import json
import numpy as np
import requests
from bs4 import BeautifulSoup
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
import urllib.request
from tqdm import tqdm

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Load sentiment model and tokenizer
def load_model_and_tokenizer(task='sentiment'):
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    
    # Download label mapping
    labels = []
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
        labels = [row[1] for row in csvreader if len(row) > 1]
    
    return model, tokenizer, labels

# Function to predict sentiment from a text
def get_sentiment_score(text, model, tokenizer, labels):
    # Preprocess text
    text = preprocess(text)
    
    # Tokenize the text
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    
    # Ensure tokenized length is within limits
    input_length = len(encoded_input['input_ids'][0])
    if input_length > 512:
        print(f"Warning: Text length exceeds model's max token limit ({input_length} tokens). Truncating.")
    
    # Predict sentiment
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    
    # Ranking the scores
    ranking = np.argsort(scores)
    ranking = ranking[::-1]  
    
    # Prepare results to save to JSON
    sentiment_results = []
    for i in range(scores.shape[0]):
        label = labels[ranking[i]]
        score = scores[ranking[i]]
        sentiment_results.append({
            "label": label,
            "score": float(score)
        })
    
    # Save results to sentiment.json
    with open("sentiment.json", "w", encoding="utf-8") as json_file:
        json.dump(sentiment_results, json_file, ensure_ascii=False, indent=4)
    
    # Return the score for the highest sentiment
    return scores[ranking[0]]

# Function to load a .txt file
def load_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    print(f"Loaded text with {len(text.split())} words")
    return text

# Download Wikipedia HTML for a product
def download_wikipedia_html(product_name):
    search_url = f"https://en.wikipedia.org/wiki/{product_name.replace(' ', '_')}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    
    response = requests.get(search_url, headers=headers)
    
    if response.status_code == 200:
        filename = f"{product_name.replace(' ', '_')}_wikipedia.html"
        with open(filename, "w", encoding="utf-8") as file:
            file.write(response.text)
        print(f"HTML saved to {filename}")
        return filename
    else:
        print(f"Failed to retrieve Wikipedia page for '{product_name}', status code: {response.status_code}")
        return None

# Find "Reception" section in Wikipedia HTML and extract its content
def find_reception_heading(html_file):
    with open(html_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    reception_heading_found = False
    reception_heading = "Reception"
    reception_start_line = None
    for i, line in enumerate(lines):
        if reception_heading in line:
            reception_heading_found = True
            reception_start_line = i
            print(f"Found 'Reception' heading at line {i + 1}: {line.strip()}")
            break
    
    if not reception_heading_found:
        print("Reception section not found in the HTML file.")
        return None
    
    # Extract content after the "Reception" heading
    reception_content = []
    for line in lines[reception_start_line + 1:]:
        if "<h2>" in line or "<h3>" in line or "<h4>" in line:  
            break
        if line.strip():  # Only add non-empty lines
            reception_content.append(line.strip())
    
    if reception_content:
        if not os.path.exists("sentiment"):
            os.makedirs("sentiment")

        txt_filename = f"sentiment/reception_{html_file.replace('.html', '')}.txt"
        with open(txt_filename, "w", encoding="utf-8") as txt_file:
            txt_file.write("\n".join(reception_content))
        print(f"Reception content saved to {txt_filename}")
        return txt_filename
    else:
        print("No content found after the Reception section.")
        return None

# Main function to process the "Nintendo Switch"
def main(product_name):
    model, tokenizer, labels = load_model_and_tokenizer(task='sentiment')
    
    print("Wikipedia Reception Scraper")
    
    print(f"\nProcessing product: {product_name}")
    
    # Step 3: Download the Wikipedia HTML
    html_file = download_wikipedia_html(product_name)
    
    if html_file:
        # Step 4: Find the Reception heading and extract the content
        txt_file = find_reception_heading(html_file)
        
        if txt_file:
            # Step 5: Load and process the text for sentiment analysis
            text = load_txt_file(txt_file)
            sentiment_score = get_sentiment_score(text, model, tokenizer, labels)
            print(f"\nSentiment score: {sentiment_score:.4f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        product_name = sys.argv[1]
        main(product_name)
    else:
        print("Usage: python script.py <product_name>")
