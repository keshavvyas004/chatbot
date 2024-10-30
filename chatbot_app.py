import tensorflow as tf
import numpy as np
import random
import pandas as pd
import json
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, GlobalMaxPooling1D, Flatten
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import string
import tkinter as tk
from tkinter import scrolledtext,Frame,Label
#importing the dataset
with open('content.json') as contents:
    data1=json.load(contents)
#getting all data to list
tags=[]
inputs=[]
responses={}
for intent in data1['intents']:
    responses[intent['tag']]=intent['responses']
    for lines in intent['input']:
        inputs.append(lines)
        tags.append(intent['tag'])

#converting to dataframe
data =pd.DataFrame({"inputs":inputs,"tags":tags})

# Print DataFrame to debug
print(data.head())
print(data.columns)


#removing punctuations
data['inputs']=data['inputs'].apply(lambda wrd:[ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['inputs']=data['inputs'].apply(lambda wrd: ''.join(wrd))
data

#tokenizing
tokenizer=Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['inputs'])
train = tokenizer.texts_to_sequences(data['inputs'])


#apply padding
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train=pad_sequences(train)

#encoding the output
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y_train=le.fit_transform(data['tags'])

input_shape=x_train.shape[1]
print(input_shape)

#define vocabulary
vocabulary=len(tokenizer.word_index)
print("number of unique word:",vocabulary)
output_length=le.classes_.shape[0]
print("output length: ",output_length)


#creating model
i = Input(shape=(input_shape,))
x=Embedding(vocabulary+1,10)(i)
x=LSTM(10,return_sequences=True)(x)
x=Flatten()(x)
x=Dense(output_length,activation="softmax")(x)
model=Model(i,x)

#compiling the model
model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

#training
train=model.fit(x_train,y_train,epochs=200)

# Function to get bot response
def get_bot_response(user_input):
    user_input = [letters.lower() for letters in user_input if letters not in string.punctuation]
    user_input = ''.join(user_input)
    input_seq = tokenizer.texts_to_sequences([user_input])
    input_seq = pad_sequences(input_seq, maxlen=input_shape)
    output = model.predict(input_seq)
    output = output.argmax()
    response_tag = le.inverse_transform([output])[0]
    return random.choice(responses[response_tag])

#creating the GUI
def send_message():
    user_input = user_entry.get()
    if user_input.strip() != "":
        user_frame=Frame(chat_window,bg="blue",bd=2)
        user_frame.pack(anchor='e',padx=5,pady=5)
        user_label=Label(user_frame,text="You:  "+user_input,bg="blue",fg="white",wraplength=250,justify="left")
        user_label.pack(fill='both',expand=True)
        
        response = get_bot_response(user_input)
        bot_frame = Frame(chat_window, bg="white", bd=2)
        bot_frame.pack(anchor='w', padx=5, pady=5)
        bot_label = Label(bot_frame, text="Bot:  " + response, bg="white", fg="black", wraplength=250, justify="left")
        bot_label.pack(fill='both', expand=True)
        user_entry.delete(0, tk.END)
        chat_window.yview(tk.END)

root = tk.Tk()
root.title("Chatbot")
root.configure(bg='black')
chat_window = scrolledtext.ScrolledText(root, state='disabled', wrap='word')
chat_window.pack(pady=10,fill='both',expand=True)
chat_window.configure(font=("Arial", 14))

user_entry = tk.Entry(root, font=("Arial", 14))
user_entry.pack(pady=10, fill='x')
user_entry.bind("<Return>", lambda event: send_message())

send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack(pady=10)

root.mainloop()

