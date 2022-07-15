[![refinery repository](https://uploads-ssl.webflow.com/61e47fafb12bd56b40022a49/62d1586ddec8452bb40c3256_sample-projects.svg)](https://github.com/code-kern-ai/refinery-sample-projects)

<img align="right" src="https://uploads-ssl.webflow.com/61e47fafb12bd56b40022a49/6200e881452a41a0d24789f3_Group%20132.svg" width="300px">

# 💬 Conversational AI
In this use case, we show you how to create training data for a [Rasa chatbot](https://github.com/RasaHQ/rasa) using [Kern *refinery*](https://github.com/code-kern-ai/refinery). You can either import the `snapshot_data.json.zip` as a snapshot in the application, or start from scratch with the `raw_data.json`.

<img src="screenshot import snapshot.png" width=500px>

*You can import the `snapshot.json.zip` on the start screen of the application (`http://localhost:4455`)*

## Labels
The goal is to create a chatbot that is capable of understanding multiple intents from questions asked by users about financial topics, e.g. frozen bank accounts. The labels of our main labeling task `intent` are:
- 401k
- account blocked
- balance
- bill due
- card declined
- card lost
- credit score
- new card
- pay bill
- spending history
- transactions

Additionally, we aim to identify card providers, which is given in a second labeling task `entities`.

As we have english texts, we can embed our data using the `distilbert-base-uncased` transformer model directly from [🤗 Hugging Face](https://huggingface.co/distilbert-base-uncased). For our `entities` labeling task, we've chosen the `en_core_web_sm` tokenizer.

## `intent` heuristics
The heuristics we build for the `intent` task are mostly consisting of distant supervisor heuristics, i.e. looking for key terms in a sentence from a lookup list. One example is `lkp_spending_history`:

```python
# from knowledge import spending_history
spending_history = [
    "recently", "spend", "spending history", "spent last week"
]

def lkp_spending_history(record):
    for term in spending_history:
        if term.lower() in record["text"].text.lower():
            return "spending history"

```
(In the app, we have a lookup list called `spending_history` which we can import from the `knowledge` module, containing all our lookup lists as Python variables).

Also, we have an active learning heuristic called `DistilbertLR`:
```python
from sklearn.linear_model import LogisticRegression

class DistilbertLR(LearningClassifier):

    def __init__(self):
        self.model = LogisticRegression()

    @params_fit(
        embedding_name = "text-classification-distilbert-base-uncased", # pick this from the options above
        train_test_split = 0.5 # we currently have this fixed, but you'll soon be able to specify this individually!
    )
    def fit(self, embeddings, labels):
        self.model.fit(embeddings, labels)

    @params_inference(
        min_confidence = 0.5,
        label_names = None # you can specify a list to filter the predictions (e.g. ["label-a", "label-b"])
    )
    def predict_proba(self, embeddings):
        return self.model.predict_proba(embeddings)
```

## `entities` heuristic
For our `entities` task, we write a regex expression:

```python
from knowledge import card_provider
import re

def find_card_providers(record):
    def regex_search(pattern, string):
        """
        some helper function to easily iterate over regex matches
        """
        prev_end = 0
        while True:
            match = re.search(pattern, string)
            if not match:
                break

            start, end = match.span()
            yield start + prev_end, end + prev_end

            prev_end += end
            string = string[end:]
            
    for provider in card_provider:
        for start, end in regex_search(
            r"({provider})".format(provider=provider), 
            record["text"].text
        ):
            span = record["text"].char_span(start, end, alignment_mode="expand")
            yield "card provider", span.start, span.end
```

Functions like this are available in our [template functions](https://github.com/code-kern-ai/template-functions) repository. Also, if you need help writing more complex functions, don't hesitate to contact us in the [forum](https://discuss.kern.ai/) or [Discord channel](https://discord.com/invite/qf4rGCEphW).

## Weak supervision and SDK
We can weakly supervise the results as we've labeled some manual reference data helping us to evaluate the heuristics, and doing so create denoised and automated labels for our tasks. That is already cool, but we still need to convert the project data into the [Rasa format](https://rasa.com/docs/rasa/nlu-training-data/). This is where we can use the `rasa` adapter from our [Python SDK](https://github.com/code-kern-ai/refinery-python). First, install the SDK via `pip install python-refinery-sdk`, and then follow along with the next section.


## Exporting to Rasa format
```python
from refinery import Client

user_name = "your-username"
password = "your-password"
project_id = "your-project-id" # can be found in the URL of the web application

client = Client(user_name, password, project_id)
# if you run the application locally, please use the following instead:
# client = Client(username, password, project_id, uri="http://localhost:4455")

from refinery.adapter import rasa

rasa.build_intent_yaml(
  client,
  "text",
  "__intent__WEAK_SUPERVISION",
  tokenized_label_task="text__entities__WEAK_SUPERVISION"
  # with the tokenized_label_task, we gain entity-level information
)
```

This will automatically create the file `data/nlu.yml` for us, which is off-the-shelf ready for training chatbots via `rasa train`:

```yml
nlu:
- intent: check_balance
  examples: |
    - how much do I have on my [savings](account) account
    - how much money is in my [checking](account) account
    - What's the balance on my [credit card account](account)
- lookup: account
  examples: |
    - savings
    - checking
    - credit card account
```

And that's it! You can now easily build chatbot data (from simple text messages up to complex messages) via Kern *refinery*, and immediately export it to the desired Rasa format with only few lines of code.

If you like what we're working on, please leave a ⭐ for [refinery](https://github.com/code-kern-ai/refinery)!
