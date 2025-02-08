import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Extract text from files
def extract_files(uploader):

    text = []

    # Create documents directory
    if not os.path.exists("documents"):
        os.mkdir("documents")

    # Extract text from each file based on the extension of the file
    for file in uploader:
        try:
            path = os.path.join("documents", file.name)

            with open(path, "wb") as writer:
                writer.write(file.getbuffer())
    
            # Load pdf documents
            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif file.name.endswith(".txt"):
                loader = TextLoader(path)

            text.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading files: {str(e)}")
    
    return text