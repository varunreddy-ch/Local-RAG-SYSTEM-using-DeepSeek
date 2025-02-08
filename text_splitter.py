import streamlit as st
from langchain.text_splitter import CharacterTextSplitter

# Split text
def text_splitter(text):
    try:
        text_splitter = CharacterTextSplitter(
            chunk_size = 1000, chunk_overlap=200, separator="\n"
        )
        splitted_text = text_splitter.split_documents(text)
        splitted_text_content = [doc.page_content for doc in splitted_text]
        # print(splitted_text)
    except Exception as e:
        st.write(f"Error while splitting text: {e}")
    
    return [splitted_text, splitted_text_content]