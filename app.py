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
    titles = [paper.get_text() for paper in papers]
    paper_links = [paper['href'] for paper in papers]
    return titles, paper_links

# Function to fetch full-text document from PubMed
def fetch_full_text(url):
    response = requests.get(f"https://pubmed.ncbi.nlm.nih.gov{url}")
    soup = BeautifulSoup(response.text, 'html.parser')
    abstract = soup.find('div', class_='abstract-content selected')
    return abstract.get_text(strip=True) if abstract else 'No full text available.'

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

# Streamlit interface
st.set_page_config(
    page_title="MedChain",
     page_icon="🏥",
)
st.title('Health Q&A Application')

user_query = st.text_input('Ask a question:')
full_texts = []  # Initialize full_texts to be used later
titles = []      # Initialize titles to be used later
paper_links = [] # Initialize paper_links to be used later

if user_query:
    if st.checkbox('Retrieve research papers'):
        titles, paper_links = fetch_research_papers(user_query)
        st.subheader('Research Papers')
        for title in titles:
            st.write(title)

        if st.button('Summarize Research Papers'):
            full_texts = [fetch_full_text(link) for link in paper_links]
            st.subheader('Summarized Texts')
            for text in full_texts:
                st.write(text)

    else:
        if full_texts:
            # Encode documents and initialize FAISS index with real documents
            document_embeddings = get_embeddings(full_texts)
            index = initialize_index(document_embeddings)
            context = retrieve_context(user_query, index, full_texts)
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context: {context}\nQuestion: {user_query}"}
            ]
            answer = generate_response(messages)
            st.write(answer)
        else:
            # Handle scenario where there are no documents and no papers have been retrieved
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Question: {user_query}"}
            ]
            answer = generate_response(messages)
            st.write(answer)
