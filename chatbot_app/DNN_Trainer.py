# Importing ML modules
import numpy
import tflearn
import nltk
import json
import random
from os import listdir
from os.path import isfile, join
from tensorflow.python.framework import ops
from nltk.stem.lancaster import LancasterStemmer
from configparser import ConfigParser

# Setting up global variables
stemmer = LancasterStemmer()
parser = ConfigParser()
parser.read('config.ini')

DataFolder = parser.get('files', 'intents_Directory')
model_name = parser.get('files', 'model_file_name')
epochs = parser.getint('ML_variables', 'epochs')
activator = parser.get('ML_variables', 'activation')

# ------PRE-PROCESSING------- #

onlyFiles = [f for f in listdir(DataFolder) if isfile(join(DataFolder, f))]

lis = []
for i in onlyFiles:
    file = "Data/" + i
    print(f"Loading {file}")
    with open(file) as f:
        data = json.load(f)
        f.close()
    lis = lis + data['intents']
data['intents'] = lis

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
net = tflearn.fully_connected(net, 32, activation=activator)
net = tflearn.fully_connected(net, 32, activation=activator)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# Training

if __name__ == "__main__":
    print("Training the model")
    model.fit(training, output, n_epoch=epochs, batch_size=8, show_metric=True)
    model.save(model_name)
else:
    print("Loading the model")
    model.load(model_name)


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
    results = model.predict([bag_of_words])[0]
    results_index = numpy.argmax(results)
    tag = labels[results_index]

    if results[results_index] > 0.75:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        return random.choice(responses)
    else:
        return "I didn't get that, try again!"
