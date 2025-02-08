# 🚀 RAG System with DeepSeek | FAISS | BM25 | Knowledge Graph


## 📌 Project Overview  
This project implements a **Retrieval-Augmented Generation (RAG)** system leveraging **DeepSeek**, FAISS, BM25, and a knowledge graph to enhance document-based question answering.  

✅  **Hybrid retrieval system** that combines **vector search (FAISS)** and **sparse retrieval (BM25)**  
✅  **Knowledge graph integration** to extract meaningful entity relationships  
✅  **DeepSeek for hypothetical document generation** to enrich query context  
✅  **Cross-Encoder Reranking** for improving retrieval accuracy  
✅  **Interactive Streamlit UI** for seamless real-time querying  

---

## 🛠️ Tech Stack  

| Category              | Tools & Frameworks |
|----------------------|------------------|
| **LLM & AI**        | DeepSeek (Ollama), LangChain |
| **Retrieval Models** | FAISS, BM25 (Okapi), Ensemble Retriever |
| **Graph-Based RAG**  | NetworkX |
| **Reranker**        | Sentence-Transformers (Cross-Encoder) |
| **Frontend**        | Streamlit |
| **Backend**         | PyTorch |

---

## 🚀 Features  

✅ **Multi-Source Retrieval** – Combines FAISS (vector search) & BM25 (sparse retrieval)  
✅ **Graph-Based Knowledge Retrieval** – Uses a knowledge graph for contextual linking  
✅ **DeepSeek-Powered Hypothetical Document Generation** – Enhances retrieval context  
✅ **Cross-Encoder Reranking** – Improves document relevance ranking  
✅ **User-Friendly Interface** – Built with Streamlit for an interactive experience  

---

## 🏗️ Installation & Setup  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/yourusername/RAG-DeepSeek.git
cd RAG-DeepSeek
```

### **2️⃣ Create a virtual Environment**  
```bash
conda create --name ragv2 python=3.12
conda activate ragv2
```

### **3️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **4️⃣ Run the Streamlit App**  
```bash
streamlit run main.py
```
---
## 🧩 How It Works  

The **RAG (Retrieval-Augmented Generation) system** follows a structured process to provide accurate responses based on document retrieval and AI-generated content.  

### **1️⃣ Upload Documents**  
- Users upload PDFs or text files via the Streamlit interface.  
- The system **extracts** text using **PyPDFLoader** or **TextLoader**.  

### **2️⃣ Generate Hypothetical Context (DeepSeek)**  
- A **hypothetical document** is generated using **DeepSeek** to enhance query understanding.  
- This helps improve retrieval by expanding user input contextually.  

### **3️⃣ Retrieve Relevant Documents**  
- The system **retrieves the most relevant documents** using:  
  ✅ **FAISS (Vector Search)** – Finds semantically similar text.  
  ✅ **BM25 (Sparse Retrieval)** – Matches based on keyword frequency.  
  ✅ **Knowledge Graph** – Identifies relationships between entities.  

### **4️⃣ Rerank Results (Cross-Encoder)**  
- Retrieved documents are **reranked using a Cross-Encoder model** for better accuracy.  
- This step ensures that the **most relevant results** appear first.  

### **5️⃣ Interactive Chat UI (Streamlit)**  
- Users enter a query, and the system fetches **top-ranked documents**.  
- The retrieved information is displayed in a **conversational format**.  
---
### **📌 Example Workflow:**  

1️⃣ User uploads a research paper.  
2️⃣ The system extracts key content and builds a **vector & keyword-based index**.  
3️⃣ When a question is asked, the system retrieves **relevant sections** from the document.  
4️⃣ The **most relevant passages** are reranked and displayed in an interactive chat format.  

---

## 📜 License  

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute this project as per the license terms.  

---

## 📬 Contact  

💡 Feel free to reach out for collaborations, discussions, or suggestions!  

📧 **Email:** vchanda1006@gmail.com  

If you find this project useful, please consider giving it a ⭐ on GitHub! 🚀  

