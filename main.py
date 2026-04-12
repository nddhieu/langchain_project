from dotenv import load_dotenv, find_dotenv
import os

env_path = find_dotenv()
print(f"DEBUG: Found .env file at: '{env_path}'")

# Load the environment variables from the found path
load_dotenv(env_path)

def main():
    print("Hello from langchain!")
    print("Your OpenAI API key is:", os.getenv("OPENAI_API_KEY"))


if __name__ == "__main__":
    main()
