import json
import random
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

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

# App
st.title('Mental Health ChatBot AI')

#Init chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# react to user input
if prompt := st.chat_input("What is up?"):
    # display user message
    st.chat_message("user").markdown(prompt)
    # add user message to chat history
    st.session_state.messages.append({"role" : "user", "content" : prompt})

    response = f"AI ChatBot: " + ChatBot(prompt)
    # display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    # add assistant reponse
    st.session_state.messages.append({"role": "assistant", "content": response})