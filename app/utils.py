import numpy as np 
import pandas as pd
import pickle
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup

import torch 
import torch as nn

from embedders.classification.contextual import TransformerSentenceEmbedder

sent_transformer = TransformerSentenceEmbedder('zhayunduo/roberta-base-stocktwits-finetuned')
model = torch.load('stock_model.pt', map_location=torch.device('cpu'))
with open('encoder.pkl', 'rb') as pkl_file:
    encoder = pickle.load(pkl_file)

def predict_sentiment(input_headlines):
    article_embedding = sent_transformer.transform(input_headlines)
    article_embedding = np.array(article_embedding)
    torch_embedding = torch.FloatTensor(article_embedding)

    scores = []
    sentiment_list = []
    for item in torch_embedding:
        prediction = model(item)
        prediction_numpy = prediction.cpu().detach().numpy().astype(int)

        sentiment_number = [np.argmax(prediction_numpy)]
        sentiment = encoder.inverse_transform(sentiment_number)
        sentiment_list.append(sentiment)
        scores.append(sentiment_number[0])

    sent_list = [list(item) for item in sentiment_list]
    sent_flatten = [x for xs in sent_list for x in xs]

    return sent_flatten, scores

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


def get_score(scores):
    max_score = len(scores) * 2
    acutal_score = sum(scores)
    sentiment_score = round((acutal_score / max_score) * 100, 2)

    return sentiment_score