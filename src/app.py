import streamlit as st

st.set_page_config(page_title="Chat with Celonis", page_icon="💬")

st.title("Chat with Celonis")

with st.sidebar:
    st.header("Settings")
    web_url = st.text_input("Website URL")