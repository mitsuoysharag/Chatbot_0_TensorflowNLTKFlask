
# coding: utf-8

#https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077

import pickle
import json
import tflearn
import numpy as np
import random
import nltk
# import tensorflow as tf
# from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
# from nltk.corpus import stopwords
stemmer = SnowballStemmer('spanish')
# stop_words = stopwords.words('spanish')


data = pickle.load(open("training_data", "rb"))
words = data['words']
classes = data['classes']
synonyms = data['synonyms']

train_x = data['train_x']
train_y = data['train_y']

with open('intents.json') as json_data:
    intents = json.load(json_data)


#Build neural network
# tf.reset_default_graph()

net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
model.load('model/model.tflearn')

# nltk.edit_distance(?)
# stop_words = stopwords.words('spanish')

def synonym(word1, word2):
    if word1 == word2:
        return True
    for s in synonyms:
        if word1 in s and word2 in s:
            return True
    return False

def clean_up_sentences(sentence):
    sentence_words = nltk.wordpunct_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentences(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if synonym(w,s):
                bag[i]=1
                if show_details:
                    print('found in bag: %s' % w)
    return np.array(bag)


ERROR_THRESHOLD = 0.3

def classify(sentence):
    results = model.predict([bow(sentence, words)])[0]
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x:x[1], reverse=True)
    
    return_list = []
    for r in results:
        return_list.append((classes[r[0]],r[1]))
    return return_list
    
def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    for result in results:
        for i in intents['intents']:
            if i['tag']==result[0]:
                if 'context_set' in i:
                    if show_details: print('context:',i['context_set'])
                    context[userID] = i['context_set']
                if not 'context_filter' in i or (userID in context and 'context_filter' in i \
                        and i['context_filter'] == context[userID]):
                    if show_details: print('tag:',i['tag'])
                    return random.choice(i['responses'])
    return 'Disculpa. No entendi tu mensaje'


context = {}

# user_input = input()            
# while(user_input!='esc'):
#     print('rating:',classify(user_input))
# #     print('Chatbot:',response(user_input, show_details=True))
#     print('Chatbot:',response(user_input))
#     user_input = input()

