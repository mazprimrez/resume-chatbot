import logging
import os
from typing import Annotated, Literal

import numpy as np
from langchain_core.messages.tool import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from utils import prompt, utils


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    
@tool
def search():
    """Get context about Mazi Prima Reza Professional & Personal information"""
    
@tool
def to_database():
    """Detect feedback or frustated response (e.g. "This is stupid" or other frustation response)
    tool will send feedback response to noSQL database"""

tools = [search, to_database]

class Agent:
    def __init__(self):
        self._llm_tools = AzureChatOpenAI(temperature=0, 
                      model='gpt-4o', 
                      api_key=os.getenv('OPENAI_API_KEY'),
                      azure_endpoint=os.getenv("API_ENDPOINT"),
                      api_version="2024-09-01-preview").bind_tools(tools)
        self.send_to_database = utils.toDatabase()
        workflow = StateGraph(State)

        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", self.tool_node)

        workflow.add_edge(START, "agent")
        workflow.add_edge("tools", "agent")

        workflow.add_conditional_edges("agent", self.maybe_route_to_tools)

        self.graph = workflow.compile()
        
        file = np.load("dataset/docs.npy", allow_pickle=True)
        context = []
        for idx in range(len(file)):
            context.append(file[idx])
        self.context_mazi = '\n'.join(context)

    # Define the function that determines whether to continue or not
    @staticmethod
    def maybe_route_to_tools(state: State) -> Literal["tools", "__end__"]:
        """Route between human or tool nodes, depending if a tool call is made."""
        if not (msgs := state.get("messages", [])):
            raise ValueError(f"No messages found when parsing state: {state}")

        msg = msgs[-1]

        if hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
            return "tools"
        else:
            return END

    def call_model(self, state: State, config: RunnableConfig):
        system_message = prompt.SYSTEM_MESSAGE
        messages = [system_message] + ["\n\n User:"] + state["messages"]

        response = self._llm_tools.invoke(messages, config)
        # We return a list, because this will get added to the existing list
        return {"messages": response}
    
    def tool_node(self, state: State) -> State:
        """Tools node to get more context on Mazi or send feedback to database when users feeling"""
        tool_msg = state.get("messages", [])[-1]
        outbound_msgs = []
            
        for tool_call in tool_msg.tool_calls:
            if tool_call["name"] == "search":
                context = "~Context: " + self.context_mazi
            elif tool_call["name"] == "to_database":
                self.send_to_database.store_to_database(tool_call["args"]["query"])
                logging.info(f"Stored feedback to database")
                context = "~Context: Succeeded to send to database"
            else:
                raise NotImplementedError(f'Unknown tool call: {tool_call["name"]}')
            
            # Record the tool results as tool messages.
            outbound_msgs.append(
                ToolMessage(
                    content=context,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outbound_msgs}