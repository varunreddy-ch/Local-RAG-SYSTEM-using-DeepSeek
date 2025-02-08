
import streamlit as st
import torch
from sentence_transformers import CrossEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# Get ranker
def get_reranker():
    try:
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
    except Exception as e:
        st.error(f"Error while loading reranker: {str(e)}")