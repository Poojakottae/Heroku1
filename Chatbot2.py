#!/usr/bin/env python
# coding: utf-8

# # Step 2: Building our chat framework

# ## Get previously built model for getting intents

# In[1]:


# things we need for NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# things we need for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random


# 
# We’ll build a simple state-machine to handle responses, using our intents model (from the previous step) as our classifier. [That’s how chatbots work](https://medium.freecodecamp.com/how-chat-bots-work-dfff656a35e2).
# 
# > A contextual chatbot framework is a classifier within a state-machine.
# 
# - we’ll un-pickle our model and documents as well as reload our intents file
# -  Remember our chatbot framework is separate from our model build — you don’t need to rebuild your model unless the intent patterns change
# - With several hundred intents and thousands of patterns the model could take several minutes to build

# In[2]:


# restore all of our data structures
import pickle
data = pickle.load(open("training_data", "rb"))
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# import our chat-bot intents file
import json
with open('Mop intents.json') as json_data:
    intents = json.load(json_data)


# In[3]:


"""""input = keras.input(shape=[len(intent_train_x[0])]) 
intent_net = input
intent_net = layers.Dense(8)(intent_net) 
intent_net = layers.Dense(8)(intent_net) 
intent_net = layers.Dense(len(intent_train_y[0], activation='softmax')(intent_net) 
intent_model = keras.Model(inputs=[input], outputs=[output])
intent_model.load('model1') """""


# In[4]:


# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')


# Util functions

# In[5]:


def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence


def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)

    return(np.array(bag))


# In[ ]:





# In[6]:


p1 = bow("what is Mopedo?", words)
print (p1)
print (classes)


# In[7]:


# load our saved model
model.load('./model.tflearn')


# 
# 
# 
# 
# 
# 
# Checking that our model gives the same result as in the previous step

# In[8]:


def get_predicted_intent(predictions):
    return classes[np.argmax(predictions)]


assert 'Mopedo' == get_predicted_intent(model.predict([p1]))


# ## Response processor

# In[9]:



# data structure to hold user context
context = {}

ERROR_THRESHOLD = 0.25


def classify(sentence):
    # generate probabilities from the model
    results = model.predict([bow(sentence, words)])[0]

    # filter out predictions below a threshold
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]

    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []

    for r in results:
        return_list.append((classes[r[0]], r[1]))

    # return tuple of intent and probability
    return return_list


def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # set context for this intent if necessar
                    return (random.choice(i['responses']))

            results.pop(0)
    else:
        return("info@")
    

# In[10]:


classify("Mopedo")


# In[ ]:


        
        
        
        
        
        
    


# In[11]:


response('Mopedo')


# In[12]:


response('What services does moepdo provide?')


# In[13]:


response('How safe is moepdo?')


# In[14]:





# In[15]:


response('payments?')


# In[16]:


response('Can I book a ride for others?')


# In[17]:


response('Mopedo city boundaries?')


# In[18]:


response('How are they charged?')


# In[20]:





# In[ ]:


context


# In[ ]:


response('I\'d like to rent a moped')


# In[ ]:


context


# In[ ]:


response('tomorrow would be great')


# haha

# In[ ]:


context


# In[ ]:


context


# In[ ]:


response('today please')


# In[ ]:


classify('today')


# In[ ]:


# clear context
response("Hi there!", show_details=True)


# In[ ]:


response('today', show_details=True)
classify('today')


# In[ ]:


response('THanks, that was fun')

