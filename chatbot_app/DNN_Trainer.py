# Importing ML modules
import numpy
import tflearn
import nltk
import json
import random
import sys
from tensorflow.python.framework import ops
from flask import Flask, render_template, request
from nltk.stem.lancaster import LancasterStemmer

# Setting up global variables
stemmer = LancasterStemmer()
app = Flask(__name__)
train = None
for i in sys.argv[1:]:
    if i.lower() == '--train':
        train = True
    else:
        train = False

# ------PRE-PROCESSING------- #

# TODO: add support for loading multiple intent files
with open('Data\intents.json') as f:
    data = json.load(f)
    f.close()

# Tokenizing
words = []
labels = []
docs_x = []
docs_y = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

# Word Stemming
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

# Bag of words
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

# ----- DEVELOPING A DNN ----- #

# Neural network Development
ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# Training

if train:
    print("Training the model")
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("Models/model.tflearn")
else:
    model.load("Models/model.tflearn")


# ----- USING THE MODEL ----- #

def get_response(inp):

    # Develop a bag of words with the input
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(inp)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    bag_of_words = numpy.array(bag)

    # Get results using the bag of words
    results = model.predict([bag_of_words])
    results_index = numpy.argmax(results)
    tag = labels[results_index]
    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
    return random.choice(responses)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(get_response(userText))


if not train:
    if __name__ == "__main__":
        app.run()
