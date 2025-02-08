import streamlit as st
from get_knowledge_graph import retrieve_from_graph
from langchain_core.documents import Document
import torch 
import ollama

torch.cuda.empty_cache()

def generate_hypothetical_document(query):
    """Generate a hypothetical document using DeepSeek."""
    prompt = f"Write a short and informative passage about: {query}"
    
    try:
        response = ollama.chat(
            model="deepseek-r1:7b",
            messages=[{"role": "user", "content": prompt}]
        )

        # print("DeepSeek Raw Response:", response)  # Debugging

        # Extract content properly
        if "message" in response and "content" in response["message"]:
            output = response["message"]["content"]
        else:
            st.write("Unexpected DeepSeek response format:", response)
            output = "DeepSeek response was not in the expected format."

    except Exception as e:
        st.write(f"DeepSeek error: {e}")
        output = "DeepSeek generation failed."

    return f"{prompt}\n{output}"



def retrieve_documents(prompt, history):
    if "retrieval_pipeline" not in st.session_state:
        with st.chat_message("assistant"):
            st.write("Please upload files before querying.....")
            return []

    full_prompt = generate_hypothetical_document(f"{history}\n{prompt}")
    # print("Extended Prompt:", full_prompt)

    # Ensure retrieval pipeline exists
    if "ensemble" not in st.session_state.retrieval_pipeline:
        st.write("Retrieval pipeline not properly initialized.")
        return []
    else:
        print("Ensemble retrieval pipeline initialized")

    # Retrieve using FAISS and BM25
    try:
        # print("retrieving docs")
        docs = st.session_state.retrieval_pipeline["ensemble"].invoke(full_prompt)
        # print("\n\n\n\n\n\n")
        # print(docs)
        if not docs:
            st.write("No relevant documents retrieved.")
            return []
    except Exception as e:
        st.write(f"Retrieval error: {e}")
        return []

    print("Extracted docs using retriever:", len(docs))

    # Graph retrieval check
    graph_output = retrieve_from_graph(prompt, st.session_state.retrieval_pipeline["knowledge_graph"])
    
    # if graph_output is None or not graph_output:
    #     st.write("Graph retrieval returned no results.")
    # else:
    #     st.write(f"Retrieved nodes: {graph_output}")

    graph_docs = [Document(page_content=node) for node in graph_output] if graph_output else []

    # Merge retrieved documents
    docs = graph_docs + docs if graph_docs else docs

    # Reranking
    if "reranker" not in st.session_state.retrieval_pipeline:
        st.write("Reranker not initialized. Returning unranked documents.")
        return docs  # Return docs without ranking

    if not docs:
        st.warning("⚠️ No relevant documents retrieved.")
        return []
    
    pairs = [[prompt, doc.page_content] for doc in docs]
    scores = st.session_state.retrieval_pipeline["reranker"].predict(pairs)

    # Sort documents based on reranking scores
    ranked_docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]

    max_contexts = st.session_state.get("max_contexts", 5)
    return ranked_docs[:max_contexts]