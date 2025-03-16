import json
import random
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

tags = []
patterns = []

import json

with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

vector = TfidfVectorizer()
patterns_scaled = vector.fit_transform(patterns)

Bot = LogisticRegression(max_iter=100000)
Bot.fit(patterns_scaled, tags)

def ChatBot(input_message):
    input_message = vector.transform([input_message])
    pred_tag = Bot.predict(input_message)[0]
    for intent in intents['intents']:
        if intent['tag'] == pred_tag:
            responses = random.choice(intent['responses'])
            return responses

input_message = input('Enter: ')
print(ChatBot(input_message))