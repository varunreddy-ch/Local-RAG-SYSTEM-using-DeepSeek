from langchain_community.vectorstores import FAISS

def get_vector_store(text_blocks, embeddings):
    return FAISS.from_documents(text_blocks, embeddings)
