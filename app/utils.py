import numpy as np 
import pandas as pd
import requests
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup

def predict_sentiment(input_headlines):
    # Get request output from the fastapi
    response = requests.post(
        "http://localhost:7531/predict", json={"text": input_headlines}
    )

    # If response is ok, split response by labels and by confidence
    if response.status_code == 200:
        predictions = response.json()
        labels = [i['label'] for i in predictions]
        confidence = [i['confidence'] for i in predictions]

        return labels, confidence
    
    else:
        print(f"Something went wrong! Response code {response.status_code}")

# Function to get news from finwiz
def get_news(ticker):
    '''
    Function to scrape news headlines from finviz, a site for finance news. Returns a list of headlines with the corresponding date.m 
    Args:
    - ticker: name of the ticker of a company (e.g. MSFT, AAPL, etc.).
    '''
    finwiz_url = 'https://finviz.com/quote.ashx?t='
    news_tables = {}

    url = finwiz_url + ticker
    req = Request(url=url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'}) 
    response = urlopen(req)    

    # Read the contents of the file into 'html'
    html = BeautifulSoup(response, features='lxml')

    # Find 'news-table' in the Soup and load it into 'news_table'
    news_table = html.find(id='news-table')

    # Add the table to our dictionary
    news_tables[ticker] = news_table

    # Read one single day of headlines for ticker 
    company = news_tables[ticker]

    # Get all the table rows tagged in HTML with <tr> into company_tr
    company_tr = company.findAll('tr')

    news_list = []
    for i, table_row in enumerate(company_tr):
        # Read the text of the element and append to list
        a_text = table_row.a.text
        news_list.append(a_text)

        # Exit after printing 4 rows of data
        if i == 10:
            break
    
    return news_list
