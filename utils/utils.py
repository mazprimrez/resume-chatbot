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
    {"role": "system", "content": "You are a helpful assisstent that knows everything about Mazi Prima Reza. She's a female. Using the information contained in the context, give a detailed answer in 1 to 5 sentences to the question. Answer in the same language as the question's language but don't translate technical terms."},
    {"role": "user", "content": """
                    Context: Notable Projects at Vidio:  
                        • Recommendation System : Designed and implemented a collaborative filtering 
                        recommendation system using cosine similarity, which improved user engagement 
                        metrics significantly.  
                        the finance sector to summarize and analyze new and updated 
                        regulat ions, thereby aiding GRC analysts in streamlining their workflow and automating 
                        analysis to inform the team faster.  
                        Notable Projects at Metrodata:  
                        • HR Chatbot Development : Led the development of a chatbot using Microsoft Azure 
                        Open AI for the HR department.  This project involved integrating NLP models to 
                        answer employee queries about company policies, resulting in a 30% reduction in HR 
                        response time.  
                        • Regulation Analysis Tool : Developed a Generative AI tool for the finance industry 
                    Question: Can you tell me about a specific project at Vidio?"""},
    {"role": "assistant", "content":"At Vidio, she developed a recommendation model using collaborative filtering with cosine similarity, achieving a notable 0.2 Recall@20 and a 10%+ click-through rate, surpassing in-house model-based recommendation systems."},
    {"role": "user", "content": prompt},
    ]

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="gpt-3.5-turbo",
    )

    return chat_completion.choices[0].message.content
