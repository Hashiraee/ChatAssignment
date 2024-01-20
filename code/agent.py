import argparse
import os

from dotenv import load_dotenv
from fake_weather_api import get_current_weather
from llama_index import (
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.agent import OpenAIAgent
from llama_index.llms import OpenAI
from llama_index.tools import FunctionTool, QueryEngineTool, ToolMetadata
from math_functions import add, divide, multiply, power, square_root, subtract

# Constants
ENV_FILE = ".env.local"
PAPERS_DIRECTORY = "./papers"
PERSIST_DIRECTORY = "./storage"
OUTPUT_DIRECTORY = "./output"
MODEL_NAME = "gpt-4-1106-preview"
TEMPERATURE = 0.2
VERBOSE = False


def load_environment_variables():
    """Load environment variables from a .env file."""
    load_dotenv(ENV_FILE)


def get_query_from_user():
    """
    Get the query from the user.

    This function uses argparse to parse command line arguments. It expects a single argument "query" which is the
    question to ask the GPT model. The function returns the query as a string.

    Returns:
        str: The user's query for the GPT model.
    """
    parser = argparse.ArgumentParser(description="Chat with GPT")
    parser.add_argument("query", type=str, help="The query to ask GPT")
    args = parser.parse_args()

    return args.query


def load_data(title: str):
    """
    Load the data from the papers directory and build the index.
    """
    dir_name = os.path.join(PAPERS_DIRECTORY, f"{title}")
    if not os.path.exists(dir_name):
        file_name = os.path.join(PAPERS_DIRECTORY, f"{title}.pdf")
        documents = SimpleDirectoryReader(input_files=[file_name]).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(
            persist_dir=os.path.join(PERSIST_DIRECTORY, title)
        )
    else:
        storage_context = StorageContext.from_defaults(
            persist_dir=os.path.join(PERSIST_DIRECTORY, title)
        )
        index = load_index_from_storage(storage_context)

    return index


def create_service_context(model_name, temperature):
    """Create a service context with default settings and OpenAI model."""
    llm = OpenAI(model=model_name, temperature=temperature)
    return ServiceContext.from_defaults(llm=llm)


def create_query_engine_tool(index_name, service_context, tool_metadata):
    """Create a QueryEngineTool for a specific index."""
    index = load_data(index_name)
    engine = index.as_query_engine(streaming=True, service_context=service_context)
    return QueryEngineTool(query_engine=engine, metadata=tool_metadata)


def create_math_tools():
    """Create FunctionTools for math operations."""
    return [
        FunctionTool.from_defaults(fn=add),
        FunctionTool.from_defaults(fn=subtract),
        FunctionTool.from_defaults(fn=multiply),
        FunctionTool.from_defaults(fn=divide),
        FunctionTool.from_defaults(fn=power),
        FunctionTool.from_defaults(fn=square_root),
    ]


def create_weather_tool():
    """Create a FunctionTool for getting the current weather."""
    return FunctionTool.from_defaults(fn=get_current_weather)


def create_agent(tools_list, llm):
    """Create an OpenAIAgent with the provided tools."""
    return OpenAIAgent.from_tools(tools_list, llm=llm, verbose=VERBOSE)


def print_response(response):
    """Print the response from the agent."""
    response_gen = response.response_gen
    for token in response_gen:
        print(token, end="")


def main():
    # Loading environment variable(s)
    load_environment_variables()

    # Creating a service context with default settings and OpenAI model
    service_context = create_service_context(MODEL_NAME, TEMPERATURE)

    # Creating the query engine tool for the Gemini index
    gemini_tool = create_query_engine_tool(
        "google_deepmind_gemini_family",
        service_context,
        ToolMetadata(
            name="gemini_paper",
            description="Provides information about the Gemini family of models. Use a detailed plain text question as input to the tool.",
        ),
    )

    # Creating the query engine tool for the Mistral index
    mistral_tool = create_query_engine_tool(
        "mistral_ai_mixtral_of_experts",
        service_context,
        ToolMetadata(
            name="mistral_paper",
            description="Provides information about the mixtral model, especially its special architecture. Use a detailed plain text question as input to the tool.",
        ),
    )

    # Getting the query from the user from the command-line
    query_str = get_query_from_user()

    # Creating the function tools (math tools) for math operations
    math_tools = create_math_tools()

    # Here, we create a weather tool that uses the fake weather API
    weather_tool = create_weather_tool()

    # Creating a list of all tools
    tools_list = math_tools + [gemini_tool, mistral_tool, weather_tool]

    # Create an OpenAIAgent with the tools
    agent = create_agent(tools_list, service_context.llm)

    # Getting the response and printing (streamed)
    response = agent.stream_chat(query_str)
    print_response(response)


if __name__ == "__main__":
    main()
