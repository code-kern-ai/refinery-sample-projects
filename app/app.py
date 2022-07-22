import numpy as np
import pandas as pd
import csv

from predictions import predict_sentiment

import streamlit as st

#input_text = st.text_input('Paste your news headline here!')

input_file = st.file_uploader('Upload your .csv file here!')

if input_file is not None:
    input_df = pd.read_csv(input_file)
    data = input_df[input_df.columns[1]].tolist()

    sent_predictions = predict_sentiment(data)

    df = pd.DataFrame({'Headline': data, 'Sentiment': sent_predictions})
    st.dataframe(df, width=1000)
