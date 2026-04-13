from dotenv import load_dotenv, find_dotenv  # Import modules for handling environment variables and finding .env files
from langchain_ollama import ChatOllama  # Import ChatOllama class from langchain_ollama module
from langchain_core.prompts import PromptTemplate  # Import PromptTemplate class from langchain_core.prompts module

# Load environment variables (.env is still found automatically at the root!): This line loads all the environment variables from the .env file into the current session.
load_dotenv(find_dotenv())

def main():
    print("Exercise 2: Ollama integration with DeepSeek-R1!")  # Print a message to indicate the purpose of this exercise.
    # Initialize the Ollama LLM
    # We use the 14b model which is perfect for an RTX 4080 Super (16GB VRAM)
    # llm = ChatOllama(model="deepseek-r1:14b", temperature=0)  # This line initializes a new instance of the ChatOllama class with the specified model and temperature.

    # prompt = PromptTemplate.from_template("What are the 3 main benefits of running LLMs locally?")  # This line creates a new instance of the PromptTemplate class from a template string.
    # chain = prompt | llm  # This line creates a chain by piping the prompt through the language model.

    # print("Generating response from deepseek-r1:14b...")  # Print a message to indicate that the response is being generated.
    # response = chain.invoke({})  # This line invokes the chain with no input, generating the response.
    # print("\nOllama Response:\n", response.content)  # Print the response from the language model.

    print("Hello from langchain!")  # Print a message to indicate that the program is running correctly.
    information = """
    Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman and entrepreneur known for his leadership of Tesla, SpaceX, X, and xAI. Musk has been the wealthiest person in the world since 2025; as of April 2026, Forbes estimates his net worth to be US$809 billion.

    Born into a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. Musk also became an American citizen in 2002.

    In 2002, Musk founded the space technology company SpaceX, becoming its CEO and chief engineer; the company has since led innovations in reusable rockets and commercial spaceflight. Musk joined the automaker Tesla as an early investor in 2004 and became its CEO and product architect in 2008; it has since become a leader in electric vehicles. In 2015, he co-founded OpenAI to advance artificial intelligence (AI) research, but later left; growing discontent with the organization's direction and leadership in the AI boom in the 2020s led him to establish xAI, which became a subsidiary of SpaceX in 2026. In 2022, he acquired the social network Twitter, implementing significant changes, and rebranding it as X in 2023. His other businesses include the neurotechnology company Neuralink, which he co-founded in 2016, and the tunneling company the Boring Company, which he founded in 2017. In November 2025, a Tesla pay package worth $1 trillion for Musk was approved, which he is to receive over 10 years if he meets specific goals.
    """  # This multi-line string contains information about Elon Musk.
    summaryTemplate = """
    Summarize the following information about Elon Musk in 3 sentences: {information}    
    """  # This template string is used to summarize the information about Elon Musk.

    summaryPromptTemplate = PromptTemplate(input_variables=["information"], template=summaryTemplate)  # This line creates a new instance of the PromptTemplate class from the summary template.

    llm = ChatOllama(temperature=0, model="deepseek-r1:14b")  # This line initializes a new instance of the ChatOllama class with the specified temperature and model.

    chain = summaryPromptTemplate | llm  # This line creates a chain by piping the prompt through the language model.

    res = chain.invoke(input={"information": information})  # This line invokes the chain with the input, generating the response.
    print("Summary:")  # Print a message to indicate that the summary is being printed.
    print(res.content)  # Print the summary from the language model.
if __name__ == "__main__":
    main()  # Call the main function when this file is run directly.
