import argparse
import os

from dotenv import load_dotenv
from llama_index import ServiceContext, SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms import OpenAI
from prompts import refined_template, text_qa_template

# Constants
ENV_FILE = ".env.local"
PAPERS_DIRECTORY = "./papers"
OUTPUT_DIRECTORY = "./output"
MODEL_NAME = "gpt-4-1106-preview"
TEMPERATURE = 0.2


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


def get_reply(index, query_str, text_qa_template, refined_template):
    """
    This function uses the provided index to generate a response to the user's query. The response is generated
    using the text_qa_template and refined_template provided.

    Args:
        index: The index to use for generating the response.
        query_str (str): The user's query.
        text_qa_template (str): The template to use for generating the response.
        refined_template (str): The template to use for refining the response.

    Returns:
        response (str): The generated response to the user's query.
    """
    response = index.as_query_engine(
        streaming=True,
        text_qa_template=text_qa_template,
        refine_template=refined_template,
    ).query(query_str)

    return response


def save_math_expression(response):
    """
    Extracts the mathematical expression from the response, writes it to a file, and prints the file location.

    If the user want to run the code (after inspecting the code), they can run the following command:
    `make calculate`

    Args:
        response (str): The response from which to extract the mathematical expression.

    Returns:
        None
    """
    code_lines = response.split("\n")[1:-1]
    code = "\n".join(code_lines)

    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    # Write the code to a file
    with open(f"{OUTPUT_DIRECTORY}/math_expression.py", "w") as file:
        file.write(code)

    print(f"\nCode written to {OUTPUT_DIRECTORY}/math_expression.py")


def main():
    # Loading environment variable(s)
    load_environment_variables()

    # Creating a service context with default settings and OpenAI model
    service_context = ServiceContext.from_defaults(
        llm=OpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
    )

    # Loading the recent (not part of training data) LLM related papers from arXiv
    documents = SimpleDirectoryReader(PAPERS_DIRECTORY).load_data()

    # Building the index
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)

    # Getting the query from the user from the command-line
    query_str = get_query_from_user()

    # Querying the index and constructing the response
    response = get_reply(index, query_str, text_qa_template, refined_template)

    # Steaming the response
    response.print_response_stream()
    full_response = str(response.get_response())

    # If the response contains code (for Mathematical expressions), we save it.
    if full_response.startswith("```python") and full_response.endswith("```"):
        save_math_expression(full_response)


if __name__ == "__main__":
    main()
