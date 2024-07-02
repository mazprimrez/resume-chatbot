from utils.utils import get_context, inference, get_document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from flask import Flask, jsonify, request

from openai import OpenAI
import os

app = Flask(__name__)
docs = get_document()
client = OpenAI(
    api_key="sk-proj-3RVXTr4Ixl6R6rfTN7clT3BlbkFJowpur0slEdi5s8L7tJe5"
)
encoder = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2', model_kwargs = {'device': "cpu"})
faiss_db = FAISS.from_documents(docs, encoder)

def get_answer(question):
    context = get_context(question, faiss_db)
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
    app.run(debug=True)
    
