import numpy as np
import pandas as pd
import csv

from utils import predict_sentiment, get_news, get_score

import streamlit as st

ticker_name = st.text_input('Select ticker name, e.g MSFT, AAPL, etc.')

if ticker_name:
    # Read news from site called finziz (10 latest news)
    web_news = get_news(ticker_name)

    # Predict sentiment for news headlines
    web_preds, web_scores = predict_sentiment(web_news)

    # Display in a pandas dataframe
    web_df = pd.DataFrame({'Headline': web_news, 'Sentiment': web_preds})
    st.dataframe(web_df, width=2000)

    web_sent_score = get_score(web_scores)
    st.metric('Sentiment score', f'{web_sent_score} %', 'Great')

input_file = st.file_uploader('Upload your .csv file here!')

if input_file is not None:
    # Read in uploaded .csv file, take first column and convert to list (needed to get predictions)
    input_df = pd.read_csv(input_file)
    file_news = input_df[input_df.columns[1]].tolist()

    # Get predictions from headlines
    file_preds, file_scores = predict_sentiment(file_news)

    # Display into a pandas dataframe
    file_df = pd.DataFrame({'Headline': file_news, 'Sentiment': file_preds})
    st.dataframe(file_df, width=2000)

    file_sent_score = get_score(file_scores)
    st.metric('Sentiment score', f'{file_sent_score} %', 'Great')


st.write('Build with streamlit by kern.ai.')
