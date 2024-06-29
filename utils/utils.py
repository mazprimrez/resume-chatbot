import numpy as np

def get_document():
    return np.load("dataset/docs.npy", allow_pickle=True)

def get_context(question, faiss_db):
    retrieved_docs = faiss_db.similarity_search(question, k=3)
    context = "".join(doc.page_content + "\n" for doc in retrieved_docs)
    return context

def inference(client, question: str, context: str):

    if context == None or context == "":
        prompt = f"""Give a detailed answer to the following question. Question: {question}"""
    else:
        prompt = f"""
            Context: {context}.
            Question: {question}"""

    messages = [
    {"role": "system", "content": "You are a helpful assisstent that knows everything about Mazi Prima Reza. She's a female. Using the information contained in the context, give a detailed answer in 1 to 3 sentence to the question. Answer in the same language as the question's language but don't translate technical terms."},
    {"role": "user", "content": prompt},
    ]

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="gpt-3.5-turbo",
    )

    return chat_completion.choices[0].message.content
