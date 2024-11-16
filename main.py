from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from langchain_core.messages import AIMessageChunk, HumanMessage

from utils.tools import Agent
load_dotenv()  # This loads environment variables from a .env file

app = Flask(__name__)
agent = Agent()
CORS(app)

@app.route('/predict', methods=['POST', 'GET'])
def main():
    if request.method == 'GET':
        return jsonify({"status": "API is running"})
    else:
        question = request.get_json()
        if question["input"]:
            return Response(generate_response(question["input"]), mimetype='text/plain')
        else:
            return jsonify({"answer": "Hello there! Do you have any questions about Mazi?"})


def generate_response(inputs):
    first = True
    for msg, metadata in agent.graph.stream({"messages": [inputs]}, stream_mode="messages"):
        if msg.content and not isinstance(msg, HumanMessage):
            if first:
                pass
            elif '~Context: ' in msg.content:
                pass
            else:
                yield msg.content

        if isinstance(msg, AIMessageChunk):
            if first:
                gathered = msg
                first = False
            else:
                gathered = gathered + msg

            if msg.tool_call_chunks:
                yield "INFO: getting information..."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
    