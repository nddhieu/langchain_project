from dotenv import load_dotenv, find_dotenv  # Import modules for handling environment variables and finding .env files

load_dotenv(find_dotenv())  # Load environment variables from .env file

from langchain.chat_models import init_chat_model  # Import function to initialize chat models
from langchain.tools import tool  # Import decorator for defining tools
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage    # Import message types for system and human messages
from langsmith import traceable  # Import decorator for tracing function calls

MAX_ITERATIONS = 5  # Define a constant for maximum iterations in the ReAct loop
MODEL = 'qwen3.5:9b'  # Define the latest model to be used for the agent

# --- Define Tools ---
@tool
def get_product_price(product: str, thought_process: str) -> float:
    """Lookup the current price of a product.
    
    Args:
        product: The name of the product.
        thought_process: Your step-by-step reasoning for why you are calling this tool.
    """
    print(f"\n[Tool Execution] Looking up price for: {product}")
    prices = {"laptop": 999.99, "smartphone": 499.99, "headphones": 199.99}
    return prices.get(product.lower(), 0.0)

@tool
def get_product_discount(price: float, discount_tier: str, thought_process: str) -> float:
    """Lookup the discount for a product based on its price and discount tier.
    
    Args:
        price: The price of the product.
        discount_tier: The tier of the discount (e.g., premium).
        thought_process: Your step-by-step reasoning for why you are calling this tool.
    """
    print(f"\n[Tool Execution] Looking up discount for price: {price}, tier: {discount_tier}")
    discounts = {
        "regular": 0.0,
        "premium": 0.1,
        "vip": 0.2
    }
    return round(price *(1- discounts.get(discount_tier.lower(), 0.0)), 2)  # Apply the discount and round to 2 decimal places discounts[discount_tier], 2)

# ---Agent looop---

@traceable
def run_agent(question: str):

    tools = [get_product_price, get_product_discount]  # List of tools available to the agent
    tool_dict = {tool.name: tool for tool in tools}  # Create a dictionary mapping tool names to tool functions
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)  # Wrap the language model with the tools for tool calling capabilities
    

    print (f"User Question: {question}\n")  # Print the user's question
    print("=== Agent Conversation ===\n")  # Print a header for the agent conversation
    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "CRITICAL: ALL of your thoughts, reasoning, and final answers MUST strictly be in English. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call get_product_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the get_product_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one.\n"

            )
        ),
        HumanMessage(content=question),
    ]

    for i in range(MAX_ITERATIONS):
        print(f"--- Iteration {i+1} ---")  # Print the current iteration number

        ai_message = llm_with_tools.invoke(messages)  # Get the AI's response based on the current conversation history
        print(f"AI Message: {ai_message.content}")  # Print the AI's message
        tool_calls = getattr(ai_message, "tool_calls", [])  # Get any tool calls made by the AI in its message
        if tool_calls:
            print("Tool Calls:")
            tool_call = tool_calls[0]
            tool_name = tool_call.get("name")  # Get the name of the tool being called
            print(f" - Tool Name: {tool_name}")
            tool_args = tool_call.get("args", {})  # Get the arguments passed to the tool
            if "thought_process" in tool_args:
                print(f"Agent Thoughts: {tool_args['thought_process']}")
            print(f" - Tool Args: {tool_args}")
            tool_call_id = tool_call.get("id")  # Get the unique ID of the tool call
            print(f" - Tool Call ID: {tool_call_id}")
  

            tool_to_use = tool_dict.get(tool_name)  # Get the actual tool function based on the tool name
            if tool_to_use is None:
                print(f"Error: Tool '{tool_name}' not found.")
                break

            observation = tool_to_use.invoke(tool_args)  # Invoke the tool with the provided arguments to get the observation
            print(f"Observation: {observation}")  # Print the observation returned by the tool
            messages.append(ai_message)  # Add the AI's message to the conversation history
            messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))  # Add the tool's observation as a ToolMessage to the conversation history
        else: 
            print(f"No tool calls. Final answer: {ai_message.content}"); 
            return ai_message.content
    print("Max iterations reached without a final answer.")  # Print a message if the maximum number of iterations is reached without obtaining a final answer
    return None

if __name__ == '__main__':
    question = "What is the price of a laptop, and how much would it be with a premium discount?"  # Define the user's question
    run_agent(question)  # Run the agent with the user's question as input

    
    
