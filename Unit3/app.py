'''
Bring everything together into a complete agent that can help host our extravagant gala.
Combine the guest information retrieval from `retriever.py`, 
and (web search, weather information, and Hub stats) tools from `tools.py` into a single powerful agent.
'''

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

from retriever import guest_info_tool
from tools import browser_search_tool, weather_info_tool, hub_stats_tool

chat = ChatOpenAI(model='gpt-4o-mini', temperature=0.0, verbose=True)
tools =  [browser_search_tool, weather_info_tool, hub_stats_tool, guest_info_tool]
chat_with_tools = chat.bind_tools(tools)

# States
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState)-> AgentState:
    return {
        'messages': [chat_with_tools.invoke(
            state['messages']
        )]
    }

# Building the graph
builder = StateGraph(AgentState)
builder.add_node('assistant', assistant)
builder.add_node('tools', ToolNode(tools))

builder.add_edge(START, 'assistant')
builder.add_conditional_edges(
    'assistant', 
    # If the latest message requires a tool, route to tools
    # Otherwise, provide a direct response
    tools_condition
)
builder.add_edge('tools', 'assistant')  # loop from tools assistant

# Compile the graph
alfred = builder.compile()
print(f'-------- Agent Compiled --------')


# Run the agent - ex 1
print(f'-------- Running Agent - example 1: Guest Info --------')
query = 'Tell me about Ada Lovelace.'
print(f'Query: {query}')
messages = [HumanMessage(content=query)]
response = alfred.invoke({
    "messages": messages
})

print("🎩 Alfred's Response:")
print(response['messages'][-1].content)

print(f'-------- Running Agent - example 2: Weather --------')
query = 'What is the weather like in New York? Will it be suitable for our fireworks display?'
print(f'Query: {query}')
messages = [HumanMessage(content=query)]
response = alfred.invoke({
    "messages": messages
})

print("🎩 Alfred's Response:")
print(response['messages'][-1].content)

print(f'-------- Running Agent - example 3: Model Stats --------')
query = 'One of our guests is from Qwen. What can you tell me about their most popular model?'
print(f'Query: {query}')
messages = [HumanMessage(content=query)]
response = alfred.invoke({
    "messages": messages
})

print("🎩 Alfred's Response:")
print(response['messages'][-1].content)

print(f'-------- Running Agent - example 4: Guest Info --------')
query = "I need to speak with 'Dr. Nikola Tesla' about recent advancements in wireless energy. Can you help me prepare for this conversation?"
print(f'Query: {query}')
messages = [HumanMessage(content=query)]
response = alfred.invoke({
    "messages": messages
})

print("🎩 Alfred's Response:")
print(response['messages'][-1].content)

print(f'-------- Running Agent - example 5: Conversation History Check --------')
query = 'Tell me about Ada Lovelace. How is she related to me?'
print(f'Query: {query}')
messages = [HumanMessage(content=query)]
response = alfred.invoke({
    "messages": messages
})

print("🎩 Alfred's Response:")
print(response['messages'][-1].content)
# Second interaction, referencing the first one
query = "What projects is she currently working on?"
print(f'Query: {query}')
messages = [HumanMessage(content=query)]
response = alfred.invoke(
    {
        "messages": response["messages"] + 
        [HumanMessage(content=query)]
    }
)

print("🎩 Alfred's Response:")
print(response['messages'][-1].content)
print('-------- Agent Running Completed --------')