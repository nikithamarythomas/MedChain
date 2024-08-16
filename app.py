import streamlit as st
from langchain_community.llms import OpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_community.retrievers import PubMedRetriever
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup
import logging
import base64

# Retrieve OpenAI API key from secrets.toml
api_key = st.secrets["openai"]["api_key"]
NUMBER_OF_MESSAGES_TO_DISPLAY = 20

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
    prompt = f"Based on the following context, provide a detailed response to the question: {question}"
    result = llm(prompt)
    return result.strip()

# Streamlit Page Configuration
st.set_page_config(
    page_title="MedChain",
    page_icon="med.jpg",
)

# Streamlit Title
st.title(":blue[Your Personal Healthcare Assistant]")

def img_to_base64(img_path):
  with open(img_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
def initialize_conversation():
    """
    Initialize the conversation history with system and assistant messages.

    Returns:
    - list: Initialized conversation history.
    """
    assistant_message = "Hello! I am Streamly. How can I assist you with Streamlit today?"

    conversation_history = [
        {"role": "system", "content": "You are a helpful health assistant, providing accurate and informative responses to user queries."}
    ]
    return conversation_history

@st.cache_data(show_spinner=False)
def on_chat_submit(chat_input):
    """
    Handle chat input submissions and interact with the OpenAI API.

    Parameters:
    - chat_input (str): The chat input from the user.
    - latest_updates (dict): The latest Streamlit updates fetched from a JSON file or API.

    Returns:
    - None: Updates the chat history in Streamlit's session state.
    """
    user_input = chat_input.strip().lower()

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = initialize_conversation()

    st.session_state.conversation_history.append({"role": "user", "content": user_input})

    try:
        assistant_reply = generate_response(chat_input, "")
        st.session_state.conversation_history.append({"role": "assistant", "content": assistant_reply})
        st.session_state.history.append({"role": "user", "content": user_input})
        st.session_state.history.append({"role": "assistant", "content": assistant_reply})

    except OpenAIError as e:
        logging.error(f"Error occurred: {e}")
        st.error(f"OpenAI Error: {str(e)}")

def initialize_session_state():
    """Initialize session state variables."""
    if "history" not in st.session_state:
        st.session_state.history = []
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

def main():
    """
    Display Streamlit updates and handle the chat interface.
    """
    initialize_session_state()

    if not st.session_state.history:
        initial_bot_message = "Hello! How can I assist you with Health related queries today?"
        st.session_state.history.append({"role": "assistant", "content": initial_bot_message})
        st.session_state.conversation_history = initialize_conversation()

    # Insert custom CSS for glowing effect
    st.markdown(
        """
        <style>
        .cover-glow {
            width: 100%;
            height: auto;
            padding: 3px;
            box-shadow: 
                0 0 5px #330000,
                0 0 10px #660000,
                0 0 15px #990000,
                0 0 20px #CC0000,
                0 0 25px #FF0000,
                0 0 30px #FF3333,
                0 0 35px #FF6666;
            position: relative;
            z-index: -1;
            border-radius: 45px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Load and display sidebar image
    img_path = "med.jpg"
    img_base64 = img_to_base64(img_path)
    if img_base64:
        st.sidebar.markdown(
            f'<img src="data:image/png;base64,{img_base64}" class="cover-glow">',
            unsafe_allow_html=True
        )
    st.sidebar.markdown("---")

    user_query = st.chat_input("Ask me about Health related queries:")
    full_texts = []  # Initialize full_texts to be used later
    titles = []      # Initialize titles to be used later
    paper_links = [] # Initialize paper_links to be used later


    if user_query:
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
            answer = on_chat_submit(user_query)
            st.write(answer)

    last_user_query = ''
    for message in st.session_state.history[-NUMBER_OF_MESSAGES_TO_DISPLAY:]:
        role = message["role"]
        avatar_image = "chatbot.jpg" if role == "assistant" else "user.png" if role == "user" else None
        if message['role'] == 'user':
            last_user_query = message['content']
        with st.chat_message(role, avatar=avatar_image):
            st.write(message["content"])

    is_retrived = False       
    if last_user_query:
        if st.sidebar.checkbox('Retrieve research papers'):     
            titles, paper_links = fetch_research_papers(last_user_query)
            st.subheader('Research Papers')
            if not titles:
                st.write('Could not find any research papers')
            else:
                is_retrived = True 
                for title in titles:
                    st.write(title)
        if is_retrived:
            if st.sidebar.button('Summarize Research Papers'):
                full_texts = [fetch_full_text(link) for link in paper_links]
                st.subheader('Summarized Texts')
                for text in full_texts:
                    st.write(summarize_document(text))


if __name__ == "__main__":
    main()