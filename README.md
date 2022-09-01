[![refinery repository](https://uploads-ssl.webflow.com/61e47fafb12bd56b40022a49/62d1586ddec8452bb40c3256_sample-projects.svg)](https://github.com/code-kern-ai/refinery-sample-projects)

<img align="right" src="https://uploads-ssl.webflow.com/61e47fafb12bd56b40022a49/6200e881452a41a0d24789f3_Group%20132.svg" width="300px">

# 💬 Conversational AI
In this use case, we show you how to create training data for a [Rasa chatbot](https://github.com/RasaHQ/rasa) using [Kern *refinery*](https://github.com/code-kern-ai/refinery). You can either import the `snapshot_data.json.zip` as a snapshot in the application, or start from scratch with the `raw_data.json`.

We also made a special video for this sample project, which you can see [here on YouTube](https://www.youtube.com/watch?v=h6xP4Kz5HJg)!

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

## Using .yml files to create a chatbot with RASA NLU

Besides the nlu.yml file, rasa also need some more files, so that we acutally get a usable chatbot. For this sample project, you need the following files:

- stories.yml
- rules.yml
- domain.yml
- endpoints.yml

This might sound a lot, but don't worry, it's not too complicated! 

First up, let's take a look at a stories. With stories, you manage the structure of the dialogues you want the conversations to have. Stories are really powerful, because they are not static. The underlying rasa rechnology is smart enough to switch between stories when needed, which allows us to create chatbots that are able to have natural and organic conversations. 

Stories have multiple steps and are started by indentifiny an intent by the user. The inents are learned by the chatbot by the nlu.yml file. An intent is usually followed by an action. If you want your chatbot to send out a response, the actions start with `utter_`. The responses are defined in the `domain.yml` file, but more on that later. Here is simple story to greet a user:
```yml
stories: 
- story: greeting the user
  steps:
  - intent: greeting
  - action: utter_greeting
```

Using this structure, we can now built a more complex story:
```yml
- story:  card_got_lost + affirm
  steps: 
  - intent: card lost
  - action: utter_ican_deactivate
  - intent: affirm
  - action: card_deactivation_form
  - active_loop: card_deactivation_form
  - slot_was_set:
    - requested_slot: birthday
  - slot_was_set:
    - requested_slot: card_number
  - slot_was_set:
    - requested_slot: null
  - active_loop: null
  - action: action_say_card_num
  - action: utter_anything_else
  - intent: deny
```

In the story you see above, we want to help to deactivate a lost or stolen card for the user. Once we identify the intent that a card was lost/stolen, we offer our help to the user. The user has to affirm this before we continue, otherwise the chatbot will switch to another story where this process is stopped (see the `stories.yml` file for this). 

To deactivate the card, we need to get some information from the customer, namely a birthdate and the card number. To do this, we are using rules. Rules are similar to stories, because they predefine steps that are taken in a conversation. Their structure is also similar in that you define steps to set them up. Rules are less flexible than stories, which is neat when you know that you want a conversation to take a certain path, like in this case to get information from the user. However, you shouldn't overuse rules.

For the rules, we are going to set up a form. A form is like a loop, that only closes once specific information is gathered. To avoid that the form is running forever, we need to actively open and close it. To open is, we write:

```yml
rules:
- rule: Activate form
  steps:
  - intent: affirm
  - action: card_deactivation_form
  - active_loop: card_deactivation_form
```

And to close it, we write another rule:
```yml
- rule: Submit form
  condition:
  # Condition that form is active.
  - active_loop: card_deactivation_form
  steps:
  # Form is deactivated
  - action: card_deactivation_form
  - active_loop: null
  - slot_was_set:
    - requested_slot: null
  - action: action_say_card_num
  - action: utter_anything_else
```
Notice that we have to set the `active_loop` to the corresponding action first and then also set it to `null` again. 

## Using custom actions

A great thing about RASA is that we can also use custom actions with it to use them in our chatbot. We can code custom action in Python. Down below, we've set up a custom action to recieve some previously stored information and print them out in a confirmation message:

```python
class ActionSayCardNum(Action):

    def name(self) -> Text:
        return "action_say_card_num"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        card_num = tracker.get_slot("card_number")
        if card_num is not 'None':
            dispatcher.utter_message(text=f"I have deactivated the card with the number: {card_num}!")
        else:
            dispatcher.utter_message(text="I don't know your card number.")

        return []
```

## Endpoints and config 

Before we can finally use the chatbot, we also need to create a file called `endpoints.yml` and add the following code to it:

```yml
action_endpoint:
  url: "hhtps://localhost:5055/webhook"
```

Because our chatbot is should be able to extract the birthday from the user in different formats, we are going to use a pre-trained EntityExtractor called Duckling. This is really easy to set up. 
All you need to do is go to the `config.yml` file and paste the following code:
```yml
pipeline:
  - name: "DucklingEntityExtractor"
    # url of the running duckling server
    url: "http://localhost:8000"
    # dimensions to extract
    dimensions: ["time", "number"]
    # allows you to configure the locale, by default the language is
    # used
    locale: "en_EN"
    # if not set the default timezone of Duckling is going to be used
    # needed to calculate dates from relative expressions like "tomorrow"
    timezone: "Europe/Berlin"
    # Timeout for receiving response from http url of the running duckling server
    # if not set the default timeout of duckling http url is set to 3 seconds.
    timeout : 3
``` 

After that, run:
```
docker run 8000:8000 rasa/duckling
```

## Starting the chatbot

Now we are ready to start up the chatbot. In a new terminal window, run `rasa run actions`. Back in the previous terminal window, you can use the chatbot by typing `rasa shell` or `rasa interactive` if you are curious to know that is happening under the hood!

And that's it! You can now easily build chatbot data (from simple text messages up to complex messages) via Kern *refinery*, and immediately export it to the desired Rasa format with only few lines of code.

If you like what we're working on, please leave a ⭐ for [refinery](https://github.com/code-kern-ai/refinery)!