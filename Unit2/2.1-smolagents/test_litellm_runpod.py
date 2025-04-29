import os
from PIL import Image
from smolagents import CodeAgent, DuckDuckGoSearchTool, VisitWebpageTool
from smolagents import HfApiModel, LiteLLMModel, OpenAIServerModel
import litellm


OPENAI_API_KEY = ''
RUNPOD_BASE_URL = 'https://copiv0isvfz56v-11434.proxy.runpod.net'


import math
from typing import Optional, Tuple
from smolagents import tool

@tool
def calculate_cargo_travel_time(
    origin_coords: Tuple[float, float],
    destination_coords: Tuple[float, float],
    cruising_speed: Optional[float] = 750.0,    # average speed for cargo planes
) -> float:
    """
    Calculate the travel time for a cargo plane between two points on Earth using Great Circle Distance.

    Args:
        origin_coords: Tuple of (latitude, longitude) for the starting point
        destination_coords: Tuple of (latitude, longitude) for the destination
        cruising_speed: Optional cruising speed of the cargo plane in km/h (defaults to 750 km/h for typical cargo planes)

    Returns:
        float: Travel time in hours

    Example:
        >>> # Chicago (41.8781° N, 87.6298° W) to Sydney (33.8688° S, 151.2093° E)
        >>> result = calculate_cargo_travel_time((41.8781, -87.6298), (-33.8688, 151.2093))
    """

    def to_radians(degrees: float) -> float:
        return degrees * (math.pi / 180)

    # Extract coordinates
    lat1, lon1 = map(to_radians, origin_coords)
    lat2, lon2 = map(to_radians, destination_coords)

    # Earth's radius in kilometers
    EARTH_RADIUS = 6371

    # Calculate great-circle distance using the haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        math.sin(dlat/2) **2 
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2) **2
    )
    c = 2 * math.asin(math.sqrt(a))
    distance = EARTH_RADIUS * c

    # Add 10% to account for non-direct routes and air traffic controls
    actual_distance = distance * 1.1

    # Calculate flight time
    # Add 1 hour for takeoff and landing procedures
    flight_time = (actual_distance / cruising_speed) + 1.0

    return round(flight_time, 2)



# model = HfApiModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", provider="together")   # HF ran otu of requests
# Use Ollama model using the LiteLLMModel API as stated here https://www.reddit.com/r/AI_Agents/comments/1iip2wx/building_a_smolagent_with_ollama_and_external/
# litellm._turn_on_debug()
litellm._turn_on_json()
model = LiteLLMModel(
    model_id = 'ollama_chat/deepseek-coder-v2:latest',
    api_base = RUNPOD_BASE_URL,
    # api_key = "ollama",
    # temperature = 0,
    # stream=False,
    # model_id = 'ollama/deepseek-r1:7b',
)


agent = CodeAgent(
    model = model,
    tools = [DuckDuckGoSearchTool(), VisitWebpageTool(), calculate_cargo_travel_time],
    additional_authorized_imports=["pandas", 'numpy'],
    max_steps = 1
)

task = """Find all Batman filming locations in the world, calculate the time to transfer via cargo plane to here (we're in Gotham, 40.7128° N, 74.0060° W), and return them to me as a pandas dataframe.
Also give me some supercar factories with the same cargo plane transfer time."""
result = agent.run(task)