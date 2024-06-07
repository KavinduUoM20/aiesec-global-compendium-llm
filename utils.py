from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st

# Load OpenAI API key from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Instantiate LLM Model
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)

def create_indexing():
    # Loading PDF Data
    loader = PyPDFLoader('./data/global-compendium-doc.pdf')
    docs = loader.load()

    # Splitting Data
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=20
    )
    docs = text_splitter.split_documents(docs)

    # Create OpenAIEmbeddings instance with API key
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    # Create a Chroma Vector DB and Save
    db = Chroma.from_documents(docs, embeddings, persist_directory='./data/')

def create_retrieval():
    # Create Prompt Template
    prompt = ChatPromptTemplate.from_template("""
    You're operating as an assistant for AIESEC. Your task is to assist users with inquiries regarding various topics covered in the AIESEC Global Compendium. Keep responses concise and relevant. If the information requested is not available in the compendium, do not provide any details.

    Question: {input}
    Context:
    {context}
    """)

    # Instantiate Document Chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Load Vector DB
    db = Chroma(persist_directory='./data/', embedding_function=OpenAIEmbeddings(api_key=OPENAI_API_KEY))

    # Create a Retriever
    retriever = db.as_retriever()

    # Create Retrieval Chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

if __name__ == "__main__":
    create_indexing()
    chain = create_retrieval()
    # Test the chain
    # print(chain.invoke({"input": "what is the Purpose of the Global Compendium?"}))
