#!/usr/bin/env python
# coding: utf-8

# In[1]:

import nltk
from flask import Flask, render_template, request
from flask import jsonify
#import xlrd
#import Chatbot2
from Chatbot2 import response


# In[2]:


app = Flask(__name__)
@app.route("/")
def index(name=None):
    return render_template('index.html',name=name)


# In[3]:



def home():
    
    return render_template("index.html")


# In[4]:


@app.route("/get", methods=['POST', 'GET'])
def getresponse():
    userText = request.args.get('msg')
    userText= userText.lower()
    response1 = str(response(userText))
    return jsonify(response1)
    
    


# In[ ]:


if __name__ == "__main__":
    app.run()


# In[ ]:





# In[ ]:




