'''
Grant Alfred access to the web, enabling him to find the latest news and global updates. 
Additionally, he’ll have access to weather data and Hugging Face hub model download statistics, 
so that he can make relevant conversation about fresh topics.
'''
from langchain_community.tools import DuckDuckGoSearchRun
import random
from langchain.tools import Tool
from huggingface_hub import list_models

def search_tool(query: str)-> str:
    """
    Retrieves search results from DuckDuckGo for a given query.
    """
    search_tool = DuckDuckGoSearchRun()
    results = search_tool.invoke(query)
    return results

browser_search_tool = Tool(
    name='browser_search',
    func=search_tool,
    description="Retrieves search results from DuckDuckGo for a given query."
)

def get_weather_info(location: str)-> str:
    """
    Fetches dummy weather information for a given location.
    """
    # dummy weather data:
    weather_conditions=[
        {'condition': 'Rainy', 'temp_c': 15},
        {'condition': 'Clear', 'temp_c': 25},
        {'condition': 'Cloudy', 'temp_c': 20},
        {'condition': 'Windy', 'temp_c': 20},
        {'condition': 'Snowy', 'temp_c': 0}
    ]

    # randomly select a condition to return
    data = random.choice(weather_conditions)
    return f"Weather in {location} is {data['condition']} with a temperature of {data['temp_c']}°C"

# initialise the tool:
weather_info_tool = Tool(
    name="get_weather_info",
    func=get_weather_info,
    description="Fetches dummy weather information for a given location."
)

def get_model_stats(author: str)-> str:
    """
    Fetches the most downloaded model stats given a specific author on the Hugging Face Hub.
    """
    try: 
        models = list(list_models(author=author, sort="downloads", direction=-1, limit=1))
        if models:
            model = models[0]
            return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
        else:
            return "No models found for author: {author}."
    except Exception as e:
        return f'Error fetching models for {author}: {str(e)}'

hub_stats_tool = Tool(
    name="get_model_stats",
    func=get_model_stats,
    description="Fetches the most downloaded model stats given a specific author on the Hugging Face Hub."
)
        
# ### Putting it all together: -- needed when testing this file independently
# from typing import TypedDict, Annotated
# from langgraph.graph.message import add_messages
# from langgraph.graph import StateGraph, START
# from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
# from langgraph.prebuilt  import ToolNode, tools_condition
# from langchain_openai import ChatOpenAI

# from dotenv import load_dotenv
# load_dotenv()

# chat = ChatOpenAI(model='gpt-4o-mini', temperature=0.0, verbose=True)
# tools = [browser_search_tool, weather_info_tool, hub_stats_tool]
# chat_with_tools = chat.bind_tools(tools)

# # Setup Agent State and nodes
# class AgentState(TypedDict):
#     messages: Annotated[list[AnyMessage], add_messages]

# def assistant(state: AgentState)-> AgentState:
#     return {
#         'messages': [chat_with_tools.invoke(
#             state['messages']
#         )]
#     }

# builder = StateGraph(AgentState)
# builder.add_node('assistant', assistant)
# builder.add_node('tools', ToolNode(tools))

# builder.add_edge(START, 'assistant')
# builder.add_conditional_edges(
#     'assistant', 
#     # If the latest message requires a tool, route to tools
#     # Otherwise, provide a direct response
#     tools_condition
# )
# builder.add_edge('tools', 'assistant')  # loop from tools assistant
# alfred = builder.compile()

# messages = [HumanMessage(
#     content='Who is Qwen and what is their most popular model?'
# )]
# response = alfred.invoke({
#     "messages": messages
# })

# print("🎩 Alfred's Response:")
# print(response['messages'][-1].content)

# Todo:
# Try implementing a tool that can be used to get the latest news about a specific topic.
