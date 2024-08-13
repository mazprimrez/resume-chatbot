import numpy as np
import faiss

def get_embedding(text, client, model="text-embedding-3-small"):
   embeddings = []
   for doc in text:
       docs = doc.replace('\n','')
       embeddings.append(client.embeddings.create(input = docs, model=model).data[0].embedding)
   return np.array(embeddings, dtype=np.float32)

def get_document():
    return np.load("dataset/docs.npy", allow_pickle=True)

def get_embedding_dataset():
    return np.load("dataset/embeddings.npy", allow_pickle=True)

def get_context(indices, docs):
    context = []
    for idx in indices:
        context.append(docs[idx])

    context = '\n'.join(context)
    return context

# Embed and index the documents
def index_documents():
    embeddings = get_embedding_dataset()
    index = faiss.IndexFlatL2(embeddings.shape[1])  # Build the index
    index.add(embeddings)  # Add vectors to the index
    return index, embeddings

# Example query function
def search_index(index, query, client, k=2):
    query_embedding = get_embedding([query], client)
    distances, indices = index.search(query_embedding, k)
    return distances, indices

def inference(client, question: str, context: str):

    if context == None or context == "":
        prompt = f"""Give a detailed answer to the following question. Question: {question}"""
    else:
        prompt = f"""
            Context: {context}.
            Question: {question}"""

    messages = [
    {"role": "system", "content": "You are Mazi Prima Reza. She's a Data Scientist. Using the information contained in the context, give a detailed answer in 1 to 3 sentences to the question. The shorter the better, but to be informative is the priority. Answer in English or Bahasa Indonesia based on the question's language but don't translate technical terms. Don't answer anything that is not related to Mazi. If the answer is not provided in the context you can use any facts in the context close to the question."},
    {"role": "user", "content": prompt},
    ]

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="gpt-4o-mini",
    )

    return chat_completion.choices[0].message.content
