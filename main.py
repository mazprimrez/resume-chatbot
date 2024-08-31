from utils.tools import professional_queries, greetings, personal_queries, feedback, route, contact
from flask import Flask, jsonify, request
from flask_cors import CORS

from openai import OpenAI
import os

from langchain_openai import ChatOpenAI
from openai import OpenAI
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function
from dotenv import load_dotenv

load_dotenv()  # This loads environment variables from a .env file
app = Flask(__name__)
CORS(app)

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

functions = [
    convert_to_openai_function(f) for f in [
        professional_queries, greetings, personal_queries, feedback, contact
    ]
]
model = ChatOpenAI(temperature=0, model='gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY')).bind(functions=functions)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
        You are an assistant for Mazi Prima Reza, a Data Scientist/AI Engineer with 3 years of experience.

        Context: Your knowledge is based solely on the information provided about Mazi.

        Scope of Answers:
        - You only answer questions related to Mazi.
        - If a question is unrelated or requests code, politely decline by stating that you only answer questions about Mazi.
        - Answer in user's language
        
        Tone:
        - Respond in a polite and fun way.
        - Sometimes, users might refer to you as Mazi. When they ask about "YOU," they are asking about Mazi. Answer as you are Mazi
        
        Important Note:
        - Do not generate Python code or any code if asked.
    """),
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
    