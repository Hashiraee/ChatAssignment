"""
This module, `prompts.py`, contains the definitions for chat prompts used in the chatbot application.
"""
from llama_index.llms import ChatMessage, MessageRole
from llama_index.prompts import ChatPromptTemplate

# Text QA Prompt
chat_text_qa_messages = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=(
            "You are GPT-4, a highly capable model answering questions for users according to the following rules:\n"
            "1. You ALWAYS answer the question, even if the context isn't helpful.\n"
            "2. When the question relates to mathematical expressions/evaluations, you DO NOT solve them yourself,"
            "you think step-by-step and ONLY provide (output) a piece of python code that can be used to solve the problem, and NOTHING else."
        ),
    ),
    ChatMessage(
        role=MessageRole.USER,
        content=(
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the question: {query_str}\n"
        ),
    ),
]
text_qa_template = ChatPromptTemplate(chat_text_qa_messages)

# Refined prompt for QA usage
chat_refined_messages = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=(
            "You are GPT-4, a highly capable model answering questions for users according to the following rules:\n"
            "1. You ALWAYS answer the question, even if the context isn't helpful.\n"
            "2. When the question relates to mathematical expressions/evaluations, you DO NOT solve them yourself,"
            "you think step-by-step and ONLY provide (output) a piece of python code that can be used to solve the problem, and NOTHING else."
        ),
    ),
    ChatMessage(
        role=MessageRole.USER,
        content=(
            "We have the opportunity to refine the original answer "
            "(only if needed) with some more context below.\n"
            "------------\n"
            "{context_msg}\n"
            "------------\n"
            "Given the new context, refine the original answer to better "
            "answer the question: {query_str}. "
            "If the context isn't useful, output the original answer again.\n"
            "Original Answer: {existing_answer}"
        ),
    ),
]
refined_template = ChatPromptTemplate(chat_refined_messages)
