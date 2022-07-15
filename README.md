[![refinery repository](https://uploads-ssl.webflow.com/61e47fafb12bd56b40022a49/62d1586ddec8452bb40c3256_sample-projects.svg)](https://github.com/code-kern-ai/refinery-sample-projects)

# 🙂😡 Sentiment analysis

**This use case is accompanied by a [YouTube video](https://www.youtube.com/watch?v=0XZLQlYSQEQ&ab_channel=KernAI).**

In this use-case we show you how you can build your own sentiment analysis classifier for stock news - completely from stratch! You'll scrape some interesting stock news from the internet to create your own dataset and then use Kern Refinery to easily and quickly label the data. 

To create the embeddings from our text data, we are going to use the `zhayunduo/roberta-base-stocktwits-finetuned` model from [Huggingface](huggingface.co).

Watch the corresponding tutorial [here on YouTube](https://www.youtube.com/watch?v=0XZLQlYSQEQ)!

<img align="right" src="https://uploads-ssl.webflow.com/61e47fafb12bd56b40022a49/62cb41d833c650bbe9c7122f_sentiment-analysis.svg">

## Labels

The goal of the sentiment classifier is to predict if the headline for a stock news headline is `Positive`, `Neutral` or `Negative`. 

## Heuristics

We can start building some function that detects whether a `Headline` contains a regular expression containing something like `"up by 20%"`, which is often times the case for stock data. We'll use this to build the following labeling functions:

```python
import re

def contains_percent(record):
    percentage_up = re.search('(up) (\d+(\.\d+)?%)', record['Headline'].text.lower())
    percentage_down = re.search('(down) (\d+(\.\d+)?%)', record['Headline'].text.lower())

    if percentage_up:
        return 'Positive'
    elif percentage_down:
        return 'Negative'

```
See [here](https://regex101.com/r/P2gzUl/1) for an explaination of the regex function. By the way, we could also have implemented this as two separate functions if we'd like to.

Next, we know that there are some key terms that might occur during labeling. One very simple instance is the occurence of famous investor **Warren Buffet**, which usually is written in a `Neutral` way. The simplest form to implement this would look as follows:

```python
def contains_buffet(record):
    if 'warren buffet' in record['Headline'].text.lower():
        return 'Neutral'
```

Alternatively, we could implement this as a [distant supervisor](https://docs.kern.ai/docs/building-labeling-functions#lookup-lists-for-distant-supervision) which looks up famous investors. This would look something like this:

```python
from knowledge import famous_investors # we'd need to create a lookup list in the app for this

def contains_investor(record):
    for investor in famous_investors:
        if investor.lower() in record['Headline'].text.lower():
            return 'Neutral'
```


Next, we're going to do something that really almost always just helps to boost your labeling: integrating an active learner. We've created embeddings using the transformer `zhayunduo/roberta-base-stocktwits-finetuned` from [🤗 Hugging Face](https://huggingface.co/zhayunduo/roberta-base-stocktwits-finetuned), so we can now implement the following:

```python
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
        min_confidence = 0.0, # we want every prediction, but we could also increase the minimum required confidence
        label_names = None # you can specify a list to filter the predictions (e.g. ["label-a", "label-b"])
    )
    def predict_proba(self, embeddings):
        return self.model.predict_proba(embeddings)

```

And that's it; from here, we can create a first version and build a simple classificator, e.g. via [automl-docker](https://github.com/code-kern-ai/automl-docker). Also, you can continue to build heuristics and doing so improve your label quantity **and** quality.

If you like what we're working on, please leave a ⭐ for [refinery](https://github.com/code-kern-ai/refinery)!
