# Chat with Celonis Docs Streamlit App

This repository contains the code for a Streamlit-based web application designed to facilitate a conversational interface with Celonis Documentation. It leverages various components from the `langchain` library to process documents and generate conversational responses based on retrieved context.

## Description

The app provides a chatbot interface where users can interact with a JR. Solution Architect bot. It retrieves document content from a specified URL, processes the text into chunks, and uses these chunks to generate contextually relevant responses to user queries. The responses are generated using a combination of history-aware retrieval chains and conversational chains with the aid of OpenAI's language models.

## Features

- Interactive chat interface using 
- Contextual document retrieval and processing
- Conversation-based document exploration
- Integration with OpenAI's language models

## Installation

To run this app locally, you will need to have Python installed on your machine. Follow these steps to set up the environment:

Clone the repository:
```bash
git clone https://github.com/your-username/chat-with-celonis.git
cd chat-with-celonis
```
Setup virtual enviornment (optional):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
Install the required packages:
```bash
pip install -r requirements.txt
```
Create your own .env file with the following variables:
```bash
OPENAI_API_KEY=[your-openai-api-key]
```

## Usage
To run the Streamlit app:
```bash
streamlit run app.py
```
