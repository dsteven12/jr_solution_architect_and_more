import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from typing import Any, List

load_dotenv()

def get_vectorstore_from_url(url: str) -> Chroma:
    """
    Creates a vector store from a document at the given URL.

    Args:
    - url (str): The URL of the document to load and process.

    Returns:
    - a vector store initialized with document chunks.
    """
    loader = WebBaseLoader(url)
    document = loader.load()

    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    # create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

    return vector_store

def get_context_retriever_chain(vector_store):
    """
    Creates a retriever chain for retrieving context-relevant documents.

    Args:
    - vector_store (Chroma): The vector store to use for retrieval.

    Returns:
    - a retrievalChain: A chain configured for history-aware document retrieval.
    """
    
    llm = ChatOpenAI()

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get the information relevant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    """
    Creates a chain for generating responses based on retrieved documents and conversation history.

    Args:
    - retriever_chain: The chain used for retrieving context.

    Returns:
    - a retrievalChain: A chain for generating conversational responses.
    """
    llm = ChatOpenAI()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions beased on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),

    ])

    stuff_docs_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_docs_chain)

def get_response(user_input: str) -> str:
    """
    Generates a response to the user input using a conversational chain.

    Args:
    - user_input (str): The user's input message.

    Returns:
    - str: The generated response.
    """
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversational_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversational_rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_query
        })
    
    return response['answer']


#app config
st.set_page_config(page_title="Chat with Celonis", page_icon="💬")
st.title("Chat with Celonis")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I'm a JR. Solution Architect. How can I help you?"),
    ]

#sidebar
with st.sidebar:
    st.header("Settings")
    web_url = st.text_input("Website URL")

if web_url is None or web_url == "":
    st.info("Please enter a website URL")

else: 
    #session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(web_url)

    #user input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    #conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)