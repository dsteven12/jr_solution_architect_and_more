import streamlit as st

st.set_page_config(page_title="Chat with Celonis", page_icon="💬")

st.title("Chat with Celonis")

with st.sidebar:
    st.header("Settings")
    web_url = st.text_input("Website URL")

user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":

    with st.chat_message("Human"):
        st.write(user_query)

    with st.chat_message("AI"):
        st.write("I don't know")
