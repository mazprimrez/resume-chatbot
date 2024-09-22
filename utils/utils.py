import numpy as np
from openai import OpenAI
import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime

class ProfessionalQuestions():
    def __init__(self):
        self.docs = None
        self.client = None
        self.context = None

    def get_docs(self):
        self.docs = np.load("dataset/docs.npy", allow_pickle=True)

    def provisioning_open_ai(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def get_context(self):
        if not self.docs:
            self.get_docs()

        self.context = []
        for idx in range(len(self.docs)):
            self.context.append(self.docs[idx])

        self.context = '\n'.join(self.context)
        return self.context

    def inference(self, question: str):
        if not self.context:
            self.get_context()

        if not self.client:
            self.provisioning_open_ai()

        prompt = f"""
            Context: {self.context}.
            Question: {question}"""

        messages = [
        {"role": "system", "content": "You are Mazi Prima Reza. She's a Data Scientist. Using the information contained in the context, give a detailed answer in 1 to 3 sentences to the question. The shorter the better, but to be informative is the priority. Answer in English or Bahasa Indonesia based on the question's language but don't translate technical terms. Don't answer anything that is not related to Mazi. If the answer is not provided in the context you can use any facts in the context close to the question. Answer in fun way, you can use emojis if needed."},
        {"role": "user", "content": prompt},
        ]

        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model="gpt-4o-mini",
            temperature = 0
        )

        return chat_completion.choices[0].message.content
    
class PersonalQuestions():
    def __init__(self):
        self.docs = None
        self.client = None
        self.context = None

    def get_docs(self):
        self.docs = np.load("dataset/facts.npy", allow_pickle=True)

    def provisioning_open_ai(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def get_context(self):
        if not self.docs:
            self.get_docs()

        self.context = []
        for idx in range(len(self.docs)):
            self.context.append(self.docs[idx])

        self.context = '\n'.join(self.context)
        return self.context

    def inference(self, question: str):
        if not self.context:
            self.get_context()

        if not self.client:
            self.provisioning_open_ai()

        messages = [
            {"role":"system", "content":"You are an assistant for Mazi Prima Reza, a 3-year experience in Data Scientist/AI Engineer. You'll help her generate response about user asking about her personal information. You can't answer the questions, but you can state one fun fact about Mazi from the context. Answer in fun way with emojis. Answer in user language."},
            {"role":"user", "content":
            f"""
            context: {self.context}
            question:{question}
            """
             }
        ]

        chat_completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=400,
            temperature=0.2,
        )

        return chat_completion.choices[0].message.content

# insert to database
class toDatabase:
    def __init__(self):
        self.client = None
        self.database_name = None
        self.collection_name = None
        self.uri = os.getenv("MONGODB_URI")

    def provision_pymongo(self):
        print('its provisioning!')
        self.client = MongoClient(self.uri, server_api=ServerApi('1'))
        print('connected!')

    def store_to_database(self, query):
        if not self.client:
            self.provision_pymongo()

        db = self.client['personal_website']
        my_collections = db['feedback_message']

        current_time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        # Data yang ingin dimasukkan
        murid_1 = {'time':current_time,'message': query}

        my_collections.insert_one(murid_1)
    