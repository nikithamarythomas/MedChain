import streamlit as st
from langchain_openai import OpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_community.retrievers import PubMedRetriever
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup

# Retrieve OpenAI API key from secrets.toml
api_key = st.secrets["openai"]["api_key"]

# Initialize OpenAI model
llm = OpenAI(api_key=api_key)

# Initialize PubMed retriever
retriever = PubMedRetriever()

# Create the prompt template for document combination
document_prompt_template = PromptTemplate.from_template(
    "Based on the following context, provide a detailed response to the question: {question}\n\nContext:\n{context}"
)

# Create the history-aware retriever chain
history_retriever_chain = create_history_aware_retriever(
    prompt=PromptTemplate.from_template("You are a helpful assistant. Based on the following history, provide an answer to the query: {input}"),
    llm=llm,
    retriever=retriever
)

# Create the document combination chain with the prompt template
document_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=document_prompt_template
)

def fetch_research_papers(query):
    url = f"https://pubmed.ncbi.nlm.nih.gov/?term={query.replace(' ', '+')}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    papers = soup.find_all('a', class_='docsum-title')
    titles = [paper.get_text() for paper in papers]
    paper_links = [paper['href'] for paper in papers]
    return titles, paper_links

def fetch_full_text(url):
    response = requests.get(f"https://pubmed.ncbi.nlm.nih.gov{url}")
    soup = BeautifulSoup(response.text, 'html.parser')
    abstract = soup.find('div', class_='abstract-content selected')
    return abstract.get_text(strip=True) if abstract else 'No full text available.'

def summarize_document(document):
    prompt = f"Please summarize the following document:\n{document}"
    result = llm(prompt)
    return result.strip()

def generate_response(question, context):
    prompt = f"Based on the following context, provide a detailed response to the question: {question}\n\nContext:\n{context}"
    result = llm(prompt)
    return result.strip()

def main():
    st.set_page_config(page_title="MedChain", page_icon="🏥")
    st.title('MedChain')

    user_query = st.text_input('Ask a question:')
    full_texts = []  # Initialize full_texts to be used later
    titles = []      # Initialize titles to be used later
    paper_links = [] # Initialize paper_links to be used later

    chat_history = []  # List to store conversation history

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
                    st.write(summarize_document(text))

        else:
            if full_texts:
                # Convert full texts to Document objects
                documents = [Document(page_content=text) for text in full_texts]
                # Generate context and answer using LangChain
                context = history_retriever_chain.invoke({
                    "input": user_query,
                    "history": [doc.page_content for doc in documents]
                })
                # Use the document chain to get the final answer
                answer = document_chain.invoke({
                    "input_documents": documents,
                    "question": user_query,
                    "context": context
                })
                st.write(answer)
            else:
                # Handle scenario where there are no documents and no papers have been retrieved
                answer = generate_response(user_query, "")
                st.write(answer)

    if chat_history:
        st.subheader('Chat History')
        for entry in chat_history:
            st.write(f"{entry['role'].capitalize()}: {entry['content']}")

if __name__ == "__main__":
    main()