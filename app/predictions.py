import numpy as np 
import pandas as pd
import pickle

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

    sentiment_list = []
    for item in torch_embedding:
        prediction = model(item)
        prediction_numpy = prediction.cpu().detach().numpy().astype(int)

        sentiment = encoder.inverse_transform([np.argmax(prediction_numpy)])
        sentiment_list.append(sentiment)

    sent_list = [list(item) for item in sentiment_list]
    sent_flatten = [x for xs in sent_list for x in xs]

    return sent_flatten