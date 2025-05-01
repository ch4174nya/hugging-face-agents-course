# RAG tool for guest stories
'''
Alfred, your trusted agent, is preparing for the most extravagant gala of the century. To ensure the event runs smoothly, Alfred needs quick access to up-to-date information about each guest. Let’s help Alfred by creating a custom Retrieval-Augmented Generation (RAG) tool, powered by our custom dataset.

**Custom Dataset**: https://huggingface.co/datasets/agents-course/unit3-invitees/

We'll create a custom tool that Alfred can use to quickly retrieve guest information during the gala. Let's break this down into three manageable steps:
1. Load and prepare the dataset
2. Create the Retriever Tool
3. Integrate the Tool with Alfred
'''
# Step 1:
# Load and prepare the dataset

import datasets
from langchain.docstore.document import Document

# Load dataset
guest_dataset = datasets.load_dataset("agents-course/unit3-invitees")

# convert dataset entries into Document objects
docs = [
    Document(
        page_content="\n".join([
            f'Name: {guest["name"]}',
            f'Relation: {guest["relation"]}',
            f'Description: {guest["description"]}',
            f'Email: {guest["email"]}'
        ]),
        metadata={"name": guest["name"]}
    )
    for guest in guest_dataset['train']
]

# Step 2:
# Create the Retriever Tool
'''
We use the `BM25Retriever` from the `langchain_community.retrievers` module to create this tool.
The `BM25Retriever` is a great starting point for retrieval, 
but for more advanced semantic search, you might consider using 
`embedding-based retrievers` like those from `sentence-transformers`.
'''

from langchain_community.retrievers import BM25Retriever
from langchain.tools import Tool

bm25_retriever = BM25Retriever.from_documents(docs)

def extract_text(query: str)-> str:
    """Retrieves detailed information about gala guest based on name or relationship"""
    results = bm25_retriever.invoke(query)

    if results:
        return '\n\n'.join([doc.page_content for doc in results[:3]])
    else:
        return "No matching guest information found."

guest_info_tool = Tool(
    name="guest_info_retriever",
    func=extract_text,
    description="Retrieves detailed information about gala guest based on name or relationship"
)

# # Step 3:
# # Integrate the Tool with Alfred  -- needed when testing this file independently

# from typing import TypedDict, Annotated
# from langgraph.graph.message import add_messages
# from langchain_core.messages import AnyMessage, HumanMessage
# from langgraph.prebuilt import ToolNode, tools_condition
# from langgraph.graph import START, StateGraph, END
# from langchain_openai import ChatOpenAI

# from dotenv import load_dotenv
# load_dotenv()

# chat = ChatOpenAI(model='gpt-4o-mini', temperature=0.0, verbose=True)
# tools = [guest_info_tool]
# chat_with_tools = chat.bind_tools(tools)

# # Building the graph

# class AgentState(TypedDict):
#     messages: Annotated[list[AnyMessage], add_messages]

# def assistant(state: AgentState) -> AgentState:
#     return {
#         'messages': [chat_with_tools.invoke(state['messages'])]
#     }

# builder = StateGraph(AgentState)

# builder.add_node('assistant', assistant)
# builder.add_node('tools', ToolNode(tools))

# builder.add_edge(START, 'assistant')
# builder.add_conditional_edges(
#     'assistant', 
#     # If the latest message requires a tool, route to tools
#     # otherwise provide a direct response
#     tools_condition
# )
# builder.add_edge('tools', 'assistant')  # loop from tools assistant

# alfred = builder.compile()

# # display the graph
# from IPython.display import Image, display
# display(Image(alfred.get_graph(xray=True).draw_mermaid_png()))

# messages = [HumanMessage(
#     content='Tell me about our guest named Ada Lovelace'
# )]
# response = alfred.invoke({
#     "messages": messages
# })

# print("Alfred's message:")
# print(response['messages'][-1].content)
