from flask import Flask, render_template, request
from flask import jsonify
#import Chatbot1
#import Chatbot2
from Chatbot2 import response

app = Flask(__name__)
@app.route("/")
def index(name=None):
    return render_template('index.html',name=name)

def home():
    return render_template("index.html")

@app.route("/get", methods=['POST', 'GET'])
def getresponse():
    userText = request.args.get('msg')
    userText= userText.lower()
    response1 = str(response(userText))
    return jsonify(response1)
    


if __name__ == "__main__":
    app.run()