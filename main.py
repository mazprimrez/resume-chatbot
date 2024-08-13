from utils.utils import search_index, inference, get_document, index_documents, get_context
from flask import Flask, jsonify, request
from flask_cors import CORS

from openai import OpenAI
import os

app = Flask(__name__)
CORS(app)

docs = get_document()
index, embeddings = index_documents()
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

def get_answer(question):
    context = get_context([0,1,2,3], docs)
    return inference(client, question=question, context=context)

@app.route('/predict', methods=['POST', 'GET'])
def main():
    if request.method == 'GET':
        return jsonify({"status": "API is running"})
    else:
        question = request.get_json()
        if question["input"]:
            answer = get_answer(question=question['input'])
        else:
            answer = "Hello there! Do you have any questions about Mazi?"
        return jsonify({'prediction': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
    