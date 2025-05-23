
import os
import pickle
import time
import streamlit as st
import langchain
from langchain_community.llms import HuggingFaceHub
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()  #  Load environment variables from .env file

#  Configure HuggingFace Hub with API token and model
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"
llm_model = HuggingFaceHub(
    repo_id='google/flan-t5-small',  
    model_kwargs={"temperature": 0.6, "max_length": 512}  
)

# Streamlit UI setup
st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

# URL input fields (3 URLs max)
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

#  Button to trigger URL processing
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "vector_index.pkl "  # File to store vector embeddings
main_placefolder = st.empty() 

#  Process URLs when button is clicked
if process_url_clicked:
    #  Step 1: Load documents from URLs
    loader = UnstructuredURLLoader(urls=urls)
    main_placefolder.text("Data loading....Started...✅✅✅")
    data = loader.load()
    
    # Step 2: Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "."],  # Split by paragraphs, lines, and sentences
        chunk_size=200,  # Size of each text chunk (in characters)
    )
    main_placefolder.text("Text Splitter....Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    
    # Step 3: Create embeddings and vector store
    embedding_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'  #  Embedding model
    )
    vectorindex_hf = FAISS.from_documents(docs, embedding_model)
    main_placefolder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)
   
    #  Step 4: Save vector index to file
    with open(file_path, "wb") as f:
        pickle.dump(vectorindex_hf, f)

# Question answering interface
query = main_placefolder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            # Load pre-built vector index
            vectorindex = pickle.load(f)
            
            # Setup QA chain with sources
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm_model,
                retriever=vectorindex.as_retriever()
            )
            result = chain({"question": query}, return_only_outputs=True)
            
            #  Display answer
            st.header("Answer")
            st.subheader(result["answer"])
            
            #  Display sources (if available)
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  #  Split multi-line sources
                for source in sources_list:
                    st.write(source)
