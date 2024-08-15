import streamlit as st
import openai
import numpy as np
import faiss
import os

# Retrieve OpenAI API key from secrets.toml
openai.api_key = st.secrets["openai"]["api_key"]


# Function to get embeddings from OpenAIstre
def get_embeddings(texts):
    response = openai.Embedding.create(
        input=texts,
        model="text-embedding-ada-002"  # Ensure this model is available
    )
    embeddings = [item['embedding'] for item in response['data']]
    return np.array(embeddings)

# Function to generate response from OpenAI (using chat model)
def generate_response(messages):
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Use a chat model
        messages=messages
    )
    return response.choices[0].message['content'].strip()

# Initialize FAISS index
def initialize_index(embeddings):
    dimension = embeddings.shape[1]  # Dimension of embeddings
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Function to retrieve relevant context from indexed documents
def retrieve_context(query, index, documents):
    query_embedding = get_embeddings([query])
    D, I = index.search(query_embedding, 1)  # Retrieve top 1 document
    top_doc_index = I[0][0]
    context = documents[top_doc_index]
    return context

# Sample documents for indexing (replace with actual documents)
documents = [
    "Document 1: Health guidelines on cardiovascular diseases.",
    "Document 2: Information about diabetes management.",
    "Document 3: Latest research on mental health treatments."
]

# Encode documents and initialize FAISS index
document_embeddings = get_embeddings(documents)
index = initialize_index(document_embeddings)

# Streamlit interface
st.title('Health Q&A Application')

user_query = st.text_input('Ask a question:')

if user_query:
    context = retrieve_context(user_query, index, documents)  # Retrieve relevant context
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Context: {context}\nQuestion: {user_query}"}
    ]
    answer = generate_response(messages)  # Generate response
    st.write(answer)