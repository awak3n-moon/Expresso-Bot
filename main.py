import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json


with open("intents.json") as file:
    data = json.load(file)



words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)


training = numpy.array(training)
output = numpy.array(output)

#tensorflow.reset_default_graph()
from tensorflow.python.framework import ops
ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

import tkinter
from tkinter import *
import pyttsx3

#engine = pyttsx3.init()
#engine.setProperty('rate', 90)

def send():
    msg = EntryBox.get("1.0",'end-1c')
    EntryBox.delete("0.0",END)   
    

    results = model.predict([bag_of_words(msg, words)])
    results_index = numpy.argmax(results)
    tag = labels[results_index]

    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
            
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

        res = random.choice(responses)
        ChatLog.insert(END, "Bot: " + res + '\n\n')
        

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('rate', 125)
        engine.setProperty('voice', voices[1].id)
        engine.say(res)
        engine.runAndWait()

        

        
base = Tk()
base.title("Expresso-Bot")
base.geometry("600x500")
base.resizable(width=FALSE, height=FALSE)


ChatLog = Text(base, bd=0, bg="grey", height="8", width="50", font="Arial",)

ChatLog.config(state=DISABLED)


scrollbar = Scrollbar(base, command=ChatLog.yview,cursor="arrow")
ChatLog['yscrollcommand'] = scrollbar.set


SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )


EntryBox = Text(base, bd=0, bg="grey",width="29", height="5", font="Arial")


scrollbar.place(x=576,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=570)
EntryBox.place(x=128, y=401, height=90, width=448)
SendButton.place(x=6, y=401, height=90)

base.mainloop()
