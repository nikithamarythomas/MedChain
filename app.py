import streamlit as st
import openai
import numpy as np
import faiss
import requests
from bs4 import BeautifulSoup

# Retrieve OpenAI API key from secrets.toml
openai.api_key = st.secrets["openai"]["api_key"]

# Function to get embeddings from OpenAI
def get_embeddings(texts):
    response = openai.Embedding.create(
        input=texts,
        model="text-embedding-ada-002"
    )
    embeddings = [item['embedding'] for item in response['data']]
    return np.array(embeddings)

# Function to generate a response from OpenAI (using chat model)
def generate_response(messages):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )
    return response.choices[0].message['content'].strip()

# Initialize FAISS index
def initialize_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Function to retrieve relevant context from indexed documents
def retrieve_context(query, index, documents):
    query_embedding = get_embeddings([query])
    D, I = index.search(query_embedding, 1)
    top_doc_index = I[0][0]
    context = documents[top_doc_index]
    return context

# Function to fetch research papers from PubMed
def fetch_research_papers(query):
    url = f"https://pubmed.ncbi.nlm.nih.gov/?term={query.replace(' ', '+')}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    papers = soup.find_all('a', class_='docsum-title')
    return [paper.get_text() for paper in papers]

# Function to summarize a document
def summarize_document(document):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Please summarize the following document:\n{document}"}
        ]
    )
    return response.choices[0].message['content'].strip()

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
    if st.checkbox('Retrieve research papers'):
        research_papers = fetch_research_papers(user_query)
        st.subheader('Research Papers')
        for paper in research_papers:
            st.write(paper)

        if st.button('Summarize Research Papers'):
            summaries = [summarize_document(paper) for paper in research_papers]
            st.subheader('Summaries')
            for summary in summaries:
                st.write(summary)
    else:
        context = retrieve_context(user_query, index, documents)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}\nQuestion: {user_query}"}
        ]
        answer = generate_response(messages)
        st.write(answer)
