from utils.tools import professional_queries, greetings, personal_queries, feedback, route
from flask import Flask, jsonify, request
from flask_cors import CORS

from openai import OpenAI
import os

from langchain_openai import ChatOpenAI
from openai import OpenAI
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function

app = Flask(__name__)
CORS(app)

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)


functions = [
    convert_to_openai_function(f) for f in [
        professional_queries, greetings, personal_queries, feedback
    ]
]
model = ChatOpenAI(temperature=0, model='gpt-4o-mini').bind(functions=functions)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant for Mazi Prima Reza, a 3-year experience in Data Scientist/AI Engineer."),
    ("user", "{input}"),
])
chain = prompt | model | OpenAIFunctionsAgentOutputParser() | route

@app.route('/predict', methods=['POST', 'GET'])
def main():
    if request.method == 'GET':
        return jsonify({"status": "API is running"})
    else:
        question = request.get_json()
        if question["input"]:
            answer = chain.invoke({"input": question["input"]})
        else:
            answer = "Hello there! Do you have any questions about Mazi?"
        return jsonify({'prediction': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
    