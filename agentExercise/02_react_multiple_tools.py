import os
from dotenv import load_dotenv, find_dotenv
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables
load_dotenv(find_dotenv())

# --- Define Multiple Tools ---

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a specific location."""
    print(f"\n[Tool Execution] 🌤️ Checking weather for: {location}")
    # In a real app, you would call a weather API here. We'll return mock data.
    return f"The weather in {location} is 75°F and sunny."

@tool
def calculate_travel_time(distance_miles: float, speed_mph: float) -> float:
    """Calculate travel time in hours given distance in miles and speed in mph."""
    print(f"\n[Tool Execution] 🚗 Calculating travel time: {distance_miles} miles at {speed_mph} mph")
    return distance_miles / speed_mph

def main():
    print("=== Exercise 2: ReAct Loop with Multiple Tools ===\n")
    
    # 1. Initialize the LLM (qwen2.5:14b is great at tool calling)
    llm = ChatOllama(model="qwen3.5:9b", temperature=0)
    
    # 2. Bundle the tools into a list
    tools = [get_weather, calculate_travel_time]
    
    # 3. Create the underlying agent using LangChain v1 syntax
    agent = create_agent(model=llm, tools=tools)
    
    # 4. Run the agent with a complex query
    query = "I am traveling to Austin, TX. What is the weather like there, and how long will it take me to drive 240 miles at 60 mph?"
    
    print(f"User Query: {query}")
    
    # 5. Invoke the agent with the conversation history
    response = agent.invoke({
        "messages": [
            SystemMessage(content="You are a helpful travel assistant. You have access to tools to help answer the user's request. "
                                  "Use them if needed. Provide a friendly final response summarizing all findings."),
            HumanMessage(content=query)
        ]
    })
    
    # The final answer is the content of the last message in the response
    final_message = response["messages"][-1]
    print("\n=== Final Response ===\n", final_message.content)

if __name__ == '__main__':
    main()