from openai import OpenAI
from langchain.agents import tool

from pydantic import BaseModel, Field
from utils.utils import ProfessionalQuestions, toDatabase

import os

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)
database = toDatabase()
model = ProfessionalQuestions()

from langchain.schema.agent import AgentFinish
def route(result):
    if isinstance(result, AgentFinish):
        return result.return_values['output']
    else:
        tools = {
            "professional_queries": professional_queries, 
            "greetings": greetings,
            "personal_queries": personal_queries,
            "feedback": feedback,
        }
        return tools[result.tool].run(result.tool_input)


# answering professional related question
class ProfessionalQueries(BaseModel):
    query: str = Field(description="Questions about professional experience")

@tool(args_schema=ProfessionalQueries)
def professional_queries(query: str) -> str:
    """function to generate answer related to Mazi's professional experience in AI and Data Scientist"""
    return model.inference(question=query)

# respoding to greetings message
class Greetings(BaseModel):
    query: str = Field(description="Greetings")

@tool(args_schema=Greetings)
def greetings(query: str) -> str:
    """Generate response for greetings."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # You can use other engines like gpt-3.5-turbo or gpt-4 if available
        messages=[
            {"role":"system", "content":"You are an assistant for Mazi Prima Reza. You'll help her generate greetings based on users input, please make the greetings fun and add emojis if needed. After responding to user greetings, you should ask them is there any questions you'd like to ask about Mazi. Answer in polite and fun way. Answer in user language."},
            {"role":"user", "content":query}
        ],
        max_tokens=100,
        temperature=0.7,
    )
    
    # Extract the generated greeting from the response
    greeting = response.choices[0].message.content
    return greeting


# answering personal related question
class PersonalQueries(BaseModel):
    query: str = Field(description="Queries about personal information")

@tool(args_schema=PersonalQueries)
def personal_queries(query: str) -> str:
    """Generate response for queries about personal information."""
    context = ProfessionalQuestions().get_context()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system", "content":"You are an assistant for Mazi Prima Reza, a 3-year experience in Data Scientist/AI Engineer. You'll help her generate response about user asking about her personal information. You can't answer the questions, but you can state one fun fact about Mazi from the context. Answer in fun way with emojis. Answer in user language."},
            {"role":"user", "content":
            f"""
            context: {context}
            question:{query}
            """
             }
        ],
        max_tokens=100,
        temperature=0.2,
    )
    
    # Extract the generated greeting from the response
    greeting = response.choices[0].message.content
    return greeting

# respoding to feedback related message
class Feedback(BaseModel):
    query: str = Field(description="Queries about feedback on chatbot")

@tool(args_schema=Feedback)
def feedback(query: str) -> str:
    """Generate response for queries about feedback and store feedback to database."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # You can use other engines like gpt-3.5-turbo or gpt-4 if available
        messages=[
            {"role":"system", "content":"You are an assistant for Mazi Prima Reza, a 3-year experience in Data Scientist/AI Engineer. You'll help her generate response for feedback about this chatbot. Say that Mazi's still learning on building this and keep this feedback in a database to be read later"},
            {"role":"user", "content":
            f"""
            question:{query}
            """
             }
        ],
        max_tokens=100,
        temperature=0.7,
    )
    
    # Extract the generated greeting from the response
    greeting = response.choices[0].message.content
    toDatabase().store_to_database(query)
    return greeting