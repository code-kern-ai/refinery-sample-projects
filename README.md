# 🙂😡 Sentiment analysis

In this use case we shot you how you can build your own sentiment anlysis classifiert for stock news - completely from stratch! You'll scrape some interesting stock news from the internet to create your own dataset and then use Kern Refinery to easily and quickly label the data. 

Watch the corresponding tutorial [here on YouTube](https://www.youtube.com/watch?v=0XZLQlYSQEQ)!

<img align="right" src="https://uploads-ssl.webflow.com/61e47fafb12bd56b40022a49/62cb41d833c650bbe9c7122f_sentiment-analysis.svg">

## Labels

The goal of the sentiment classifier is to predict if the headline for a stock news is positive, neutral or negative. 

To create the embeddings from our text, we are going to use the 'zhayunduo/roberta-base-stocktwits-finetuned' model from [Huggingface](huggingface.co).

## Heuristics

Labeling function #1:
'''
import re

def contains_percent(record):
    percentage_up = re.search('(up) (\d+(\.\d+)?%)', record['Headline'].text.lower())
    percentage_down = re.search('(down) (\d+(\.\d+)?%)', record['Headline'].text.lower())

    if percentage_up:
        return 'Positive'
    elif percentage_down:
        return 'Negative'

'''
See [here](https://regex101.com/r/P2gzUl/1) for an explaination of the REGEX function. 

Labeling function #2:
'''
def contains_buffet(record):
    if 'warren buffet' in record['Headline'].text.lower():
        return 'Neutral'
'''

Active learner:
'''
from sklearn.linear_model import LogisticRegression
# you can find further models here: https://scikit-learn.org/stable/supervised_learning.html#supervised-learning

class MyActiveLearner(LearningClassifier):

    def __init__(self):
        self.model = LogisticRegression()

    @params_fit(
        embedding_name = "Headline-classification-zhayunduo/roberta-base-stocktwits-finetuned", # pick this from the options above
        train_test_split = 0.5 # we currently have this fixed, but you'll soon be able to specify this individually!
    )
    def fit(self, embeddings, labels):
        self.model.fit(embeddings, labels)

    @params_inference(
        min_confidence = 0.0,
        label_names = None # you can specify a list to filter the predictions (e.g. ["label-a", "label-b"])
    )
    def predict_proba(self, embeddings):
        return self.model.predict_proba(embeddings)

'''