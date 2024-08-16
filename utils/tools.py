
from langchain_openai import ChatOpenAI
from openai import OpenAI
from langchain_core.messages import AIMessage
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.agents import tool
from langchain_core.utils.function_calling import convert_to_openai_function

from pydantic import BaseModel, Field
from utils.utils import ProfessionalQuestions

import os

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

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


class ProfessionalQueries(BaseModel):
    query: str = Field(description="Questions about professional experience")

@tool(args_schema=ProfessionalQueries)
def professional_queries(query: str) -> str:
    """function to generate answer related to Mazi's professional experience in AI and Data Scientist"""
    return ProfessionalQuestions.inference(question=query)

class Greetings(BaseModel):
    query: str = Field(description="Greetings")

@tool(args_schema=Greetings)
def greetings(query: str) -> str:
    """Generate response for greetings."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # You can use other engines like gpt-3.5-turbo or gpt-4 if available
        messages=[
            {"role":"system", "content":"You are an assistant for Mazi Prima Reza, a 3-year experience in Data Scientist/AI Engineer. You'll help her generate greetings based on users input, please make the greetings fun and add emojis if needed. After responding to user greetings, you should ask them is there any questions you'd like to ask about Mazi in fun way."},
            {"role":"user", "content":query}
        ],
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.7,
    )
    
    # Extract the generated greeting from the response
    greeting = response.choices[0].message.content
    return greeting

class PersonalQueries(BaseModel):
    query: str = Field(description="Queries about personal information")

@tool(args_schema=PersonalQueries)
def personal_queries(query: str) -> str:
    """Generate response for queries about personal information."""
    context = ProfessionalQuestions().get_context()

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # You can use other engines like gpt-3.5-turbo or gpt-4 if available
        messages=[
            {"role":"system", "content":"You are an assistant for Mazi Prima Reza, a 3-year experience in Data Scientist/AI Engineer. You'll help her generate response about user asking about her personal information. You can't answer the questions, but you can state one fun fact about Mazi from the context. Answer in fun way with emojis."},
            {"role":"user", "content":
            f"""
            context: {context}
            question:{query}
            """
             }
        ],
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )
    
    # Extract the generated greeting from the response
    greeting = response.choices[0].message.content
    return greeting

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
        n=1,
        stop=None,
        temperature=0.7,
    )
    
    # Extract the generated greeting from the response
    greeting = response.choices[0].message.content
    return greeting