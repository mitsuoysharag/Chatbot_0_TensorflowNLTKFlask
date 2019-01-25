
# coding: utf-8

# In[1]:


#https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077

#NLP
import nltk
# from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

#Tensorflow
import numpy as np
import tflearn
#pip install tensorflow==1.5
import tensorflow as tf
import random

#File
import json
import pickle


# In[42]:


with open('intents.json') as json_data:
    intents = json.load(json_data)

stemmer = SnowballStemmer('spanish')
stop_words = [
    '¿','?',
    'que','como','cuando','donde','porque',
    'en','un','te','de','una','sus'
]
# stop_words = stopwords.words('spanish')+['¿','?']

synonyms = []
words = []
classes = []
documents = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.wordpunct_tokenize(pattern)
        words.extend(w)
        documents.append((w,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

for synonym in intents['synonyms']:
    aux = []
    for s in intents['synonyms'][synonym]:
        aux.append(stemmer.stem(s.lower()))
    synonyms.append(aux)
            
stop_words = [stemmer.stem(sw.lower()) for sw in stop_words]
words = [stemmer.stem(w.lower()) for w in words]
words = [w for w in words if w not in stop_words and len(w)>1]
    
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "words", words)
print(len(synonyms), "synonyms", synonyms)


# In[43]:


#Training data
training = []
output = []

output_empty = [0]*len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])
    
random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])

print(train_x)
print(train_y)


# In[48]:


tf.reset_default_graph()

net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model/model.tflearn')


# In[47]:


pickle.dump({'words':words, 'classes':classes, 'synonyms':synonyms, 'train_x':train_x, 'train_y':train_y}, open("training_data", "wb"))