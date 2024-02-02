import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_response(user_input):
    return "I don't know"

def get_vectorstore_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()

    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    return document_chunks

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
    document_chunks = get_vectorstore_from_url(web_url)
    with st.sidebar:
        st.write(document_chunks)
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