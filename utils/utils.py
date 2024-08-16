import numpy as np
from openai import OpenAI
import os

class ProfessionalQuestions():
    def __init__(self):
        self.docs = np.load("dataset/docs.npy", allow_pickle=True)
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.context = None

    def get_context(self):
        self.context = []
        for idx in range(len(self.docs)):
            self.context.append(self.docs[idx])

        self.context = '\n'.join(self.context)
        return self.context

    def inference(self, client, question: str):
        if not self.context:
            self.get_context()

        if self.context == None or self.context == "":
            prompt = f"""Give a detailed answer to the following question. Question: {question}"""
        else:
            prompt = f"""
                Context: {self.context}.
                Question: {question}"""

        messages = [
        {"role": "system", "content": "You are Mazi Prima Reza. She's a Data Scientist. Using the information contained in the context, give a detailed answer in 1 to 3 sentences to the question. The shorter the better, but to be informative is the priority. Answer in English or Bahasa Indonesia based on the question's language but don't translate technical terms. Don't answer anything that is not related to Mazi. If the answer is not provided in the context you can use any facts in the context close to the question."},
        {"role": "user", "content": prompt},
        ]

        chat_completion = client.chat.completions.create(
            messages=messages,
            model="gpt-4o-mini",
            temperature = 0
        )

        return chat_completion.choices[0].message.content
