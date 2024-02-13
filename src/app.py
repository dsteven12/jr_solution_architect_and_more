import streamlit as st  # Importing the Streamlit library for creating web applications
from langchain_core.messages import AIMessage, HumanMessage  # Importing message classes for AI and human messages
from langchain_community.document_loaders import WebBaseLoader  # Importing a document loader for web-based documents
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Importing a text splitter for dividing documents into chunks
from langchain_community.vectorstores import Chroma  # Importing Chroma for storing document vectors
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # Importing OpenAI tools for embeddings and chat functionalities
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # Importing tools for creating chat prompts and placeholders
from langchain.chains import create_history_aware_retriever, create_retrieval_chain  # Importing functions for creating retrieval chains
from langchain.chains.combine_documents import create_stuff_documents_chain  # Importing a function to create a chain for combining documents
from dotenv import load_dotenv  # Importing a function to load environment variables from a .env file
from typing import Any, List  # Importing typing tools for type annotations

load_dotenv()

def get_vectorstore_from_url(url: str) -> Chroma:
    """
    Creates a vector store from a document at the given URL.

    Args:
    - url (str): The URL of the document to load and process.

    Returns:
    - a vector store initialized with document chunks.
    """
    loader = WebBaseLoader(url)  # Initializing a loader for web-based documents
    document = loader.load()  # Loading the document from the specified URL

    text_splitter = RecursiveCharacterTextSplitter()  # Initializing a text splitter for document chunks
    document_chunks = text_splitter.split_documents(document)  # Splitting the document into smaller chunks

    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())  # Creating a vector store from document chunks with OpenAI embeddings

    return vector_store  # Returning the created vector store

def get_context_retriever_chain(vector_store):
    """
    Creates a retriever chain for retrieving context-relevant documents.

    Args:
    - vector_store (Chroma): The vector store to use for retrieval.

    Returns:
    - a retrievalChain: A chain configured for history-aware document retrieval.
    """
    
    llm = ChatOpenAI()  # Initializing a language model for chat-based interactions

    retriever = vector_store.as_retriever()  # Creating a retriever from the vector store

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get the information relevant to the conversation")
    ])  # Setting up a prompt template for retrieval based on conversation history

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)  # Creating a retriever chain that is aware of conversation history

    return retriever_chain  # Returning the created retriever chain

def get_conversational_rag_chain(retriever_chain):
    """
    Creates a chain for generating responses based on retrieved documents and conversation history.

    Args:
    - retriever_chain: The chain used for retrieving context.

    Returns:
    - a retrievalChain: A chain for generating conversational responses.
    """
    llm = ChatOpenAI()  # Initializing a language model for chat-based interactions, again

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])  # Setting up a prompt template for generating responses based on retrieved documents and conversation history

    stuff_docs_chain = create_stuff_documents_chain(llm, prompt)  # Creating a chain for combining documents based on the prompt

    return create_retrieval_chain(retriever_chain, stuff_docs_chain)  # Returning a chain for generating conversational responses

def get_response(user_input: str) -> str:
    """
    Generates a response to the user input using a conversational chain.

    Args:
    - user_input (str): The user's input message.

    Returns:
    - str: The generated response.
    """
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)  # Getting the context retriever chain with the current vector store
    conversational_rag_chain = get_conversational_rag_chain(retriever_chain)  # Getting the conversational chain for response generation

    response = conversational_rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_input
        })  # Invoking the conversational chain with chat history and user input to generate a response
    
    return response['answer']  # Returning the generated response


# App configuration and UI setup
st.set_page_config(page_title="Chat with Celonis", page_icon="💬")  # Setting the page configuration with title and icon
st.title("Chat with Celonis")  # Setting the page title

if "chat_history" not in st.session_state:  # Checking if chat history is not already in session state
    st.session_state.chat_history = [  # Initializing chat history in session state
        AIMessage(content="Hello, I'm a JR. Solution Architect. I'll help you learn more about the Web URL you entered above. Let's Chat It Up!"),
    ]

# Sidebar for settings
with st.sidebar:
    st.header("Settings")  # Adding a header to the sidebar
    web_url = st.text_input("Website URL", value=st.session_state.get('web_url', ''), help="Please input the URL that you'd like to chat about.")  # Text input for website URL

if web_url is None or web_url == "":
    st.info("Please enter a website URL")  # Displaying a message if the URL is not entered

else: 
    st.text(f"URL Entered: {web_url}")  # Displaying the entered URL
    if "vector_store" not in st.session_state:  # Checking if vector store is not already in session state
        st.session_state.vector_store = get_vectorstore_from_url(web_url)  # Initializing vector store in session state with the entered URL

    user_query = st.chat_input("Type your message here...")  # Chat input for user message
    if user_query is not None and user_query != "":  # Checking if user query is not empty
        response = get_response(user_query)  # Getting response to user query
        st.session_state.chat_history.append(HumanMessage(content=user_query))  # Adding user query to chat history
        st.session_state.chat_history.append(AIMessage(content=response))  # Adding AI response to chat history

    for message in st.session_state.chat_history:  # Iterating through chat history
        if isinstance(message, AIMessage):  # Checking if message is an AI message
            with st.chat_message("AI"):  # Creating an AI chat message block
                st.write(message.content)  # Displaying AI message content
        elif isinstance(message, HumanMessage):  # Checking if message is a human message
            with st.chat_message("Human"):  # Creating a human chat message block
                st.write(message.content)  # Displaying human message content