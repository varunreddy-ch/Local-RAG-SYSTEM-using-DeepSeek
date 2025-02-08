import os
import streamlit as st
import torch
from get_ranker import get_reranker
from extract_text import extract_files
from text_splitter import text_splitter
from get_embeddings import get_embeddings
from get_vector_store import get_vector_store
from get_retriever import get_retriever
from get_knowledge_graph import get_knowledge_graph
from prompt_retriever import retrieve_documents

st.set_page_config("RAG using DeepSeek", layout="wide")
device = "cuda" if torch.cuda.is_available() else "cpu"
# Custom CSS
st.markdown("""
    <style>
        .stApp { background-color: #bbc5aa; }
        h1 { color: #A72608; text-align: center; }
        .stChatMessage { border-radius: 10px; padding: 10px; margin: 10px 0; }
        .stChatMessage.user { background-color: #e8f0fe; }
        .stChatMessage.assistant { background-color: #d1e7dd; }
        .stButton>button { background-color: #090C02; color: white; }
    </style>
""", unsafe_allow_html=True)




if "messages" not in st.session_state:
    st.session_state.messages = []
if "retrieval_pipeline" not in st.session_state:
    st.session_state.retrieval_pipeline = None
if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = False
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False


final_text = None

with st.sidebar:
    st.header("Documents")
    uploader = st.file_uploader("Upload Documents", accept_multiple_files=True)

    # if uploader and not st.session_state.documents_loaded:
    if uploader:
        with st.spinner("Analyzing Documents"):
            # Get ranker
            reranker = get_reranker()
            print("reranking done")
            # Extract text from files
            extracted_text = extract_files(uploader)
            print("extracting text done")
            # Split text
            text_blocks, text_content = text_splitter(extracted_text)
            print("text splitting done")
            # Embedding for retrieval
            embeddings = get_embeddings()
            print("embedding done")
            # Vector store for storing data
            vector_store = get_vector_store(text_blocks, embeddings)
            print("vector storing done")
            # Get Retrieval
            retriever = get_retriever(text_content, vector_store)
            print("bm25 retriever done")
            # Build a knowledge graph
            knowledge_graph = get_knowledge_graph(text_blocks)
            print("buliding graph done")

            # Store the entire information in streamlit session
            st.session_state.retrieval_pipeline = {
                "ensemble": retriever,
                "reranker" : reranker,
                "text": text_content,
                "knowledge_graph": knowledge_graph
            }

            # if "knowledge_graph" in st.session_state.retrieval_pipeline:
            #     G = st.session_state.retrieval_pipeline["knowledge_graph"]
            #     st.write(f"🔗 Total Nodes: {len(G.nodes)}")
            #     st.write(f"🔗 Total Edges: {len(G.edges)}")
            #     st.write(f"🔗 Sample Nodes: {list(G.nodes)[:10]}")
            #     st.write(f"🔗 Sample Edges: {list(G.edges)[:10]}")
            st.session_state.documents_loaded = True
            st.success("Documents Analysed")
    

    st.markdown("---")
    st.header("Settings")

    # Set model parameters
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.01)
    st.session_state.max_contexts = st.slider("Max Contexts", 1, 10, 3, 1)

    st.session_state.messages = []
    # Clear everything
    if st.button("Reset"):
        st.session_state.messages = []
        st.rerun()


# Main body 
st.title("RAG system using DeepSeek")

# Chat Inteface
# st.session_state.messages = [{"role": "robo", "content": "hello world"}]
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if (prompt := st.chat_input("How can I help you...")):
    # Make an history object
    history = ""
    for message in st.session_state.messages[-5:]:
        history += message["content"]
        history += "\n"

    # Add user query into messages
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Add the user propmt into UI
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        # st.write(prompt)
        context = ""

        try:
            # st.write("processing")
            docs = retrieve_documents(prompt, history)
            for doc in docs:
                print(doc)
                print("\n\n\n")
            # st.write("done processing")
            
            for i, doc in enumerate(docs):
                source_no = f"Source {i+1}:"
                context += f"**{source_no}** \n{doc.page_content}\n\n"
        except Exception as e:
            st.error(f"Retrieval error: {str(e)}")

        response_placeholder.markdown(context)
        # Save assistant response
        st.session_state.messages.append({"role": "assistant", "content": context})

