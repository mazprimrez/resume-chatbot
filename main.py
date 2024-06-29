from utils.utils import get_context, inference, get_document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from flask import Flask, request

from openai import OpenAI
import os

app = Flask(__name__)
docs = get_document()
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
encoder = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2', model_kwargs = {'device': "cpu"})
faiss_db = FAISS.from_documents(docs, encoder)

def get_answer(question):
    context = get_context(question, faiss_db)
    return inference(client, question=question, context=context)

@app.route('/', methods=['POST'])
def main():
    question = request.get_json()
    return get_answer(question=question)

if __name__ == '__main__':
    app.run(debug=True)
    
