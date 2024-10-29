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

#plotting model accuracy
plt.plot(train.history['accuracy'], label='training set accuracy')
plt.plot(train.history['loss'],label="set loss")
plt.legend()

#chatting
while True:
    text_p=[]
    user_input = input("You: ")

    #removing punctuation and converting to lowercase
    prediction_input=[letters.lower() for letters in user_input if letters not in string.punctuation]
    prediction_input=''.join(prediction_input)
    text_p.append(prediction_input)

    #tokenizing and padding
    prediction_input = tokenizer.texts_to_sequences(text_p)
    prediction_input=np.array(prediction_input).reshape(-1)
    prediction_input=pad_sequences([prediction_input],input_shape)

    #getting output
    output=model.predict(prediction_input)
    output=output.argmax()

    #finding the right tag
    responses_tag=le.inverse_transform([output])[0]
    print("MediHealth : ",random.choice(responses[responses_tag]))
    if responses_tag=="goodbye":
        break
