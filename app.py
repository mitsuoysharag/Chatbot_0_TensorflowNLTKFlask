from flask import Flask
from flask import jsonify
from flask_cors import CORS
from ChatbotResponse import response

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return "Hola Mundo"

@app.route("/chatbot/<string:sentence>/")
def chatbot(sentence):
    return jsonify(
        message=response(sentence)
        # message=sentence    
    )

if __name__ == "__main__":
    app.run()