import os
import streamlit as st
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


@st.cache_resource(show_spinner="Connecting to embedding service...")
def _load_embeddings():
    """
    Uses the HuggingFace Inference API — the model runs on HF's servers,
    not locally. This means zero local RAM is used for the embedding model,
    solving the 512 MB OOM on Render's free tier.

    Required env var:
        HUGGINGFACEHUB_API_TOKEN  — free token from huggingface.co/settings/tokens
    """
    return HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
    )


def create_chatbot_engine(full_text):
    # ── 1. Text splitting ────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([full_text])

    # ── 2. Embeddings via HuggingFace Inference API (no local model load) ────
    embeddings = _load_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    # ── 3. Retriever ─────────────────────────────────────────────────────────
    retriever = vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={'k': 6}
    )

    # ── 4. LLM (Groq cloud — also uses zero local RAM) ───────────────────────
    # Local alternative — uncomment to run fully offline via Ollama:
    # from langchain_community.chat_models import ChatOllama
    # llm = ChatOllama(model="phi", temperature=0.3)
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # ── 5. Prompt ─────────────────────────────────────────────────────────────
    prompt = PromptTemplate(
        template="""
            You are a helpful assistant.
            Answer ONLY using the provided transcript context.
            If the context does not contain the answer, say "I don't know."

            Context:
            {context}

            Question:
            {question}

            Answer:
            """,
        input_variables=["context", "question"]
    )

    # ── 6. Chain ─────────────────────────────────────────────────────────────
    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    parser = StrOutputParser()
    return parallel_chain | prompt | llm | parser
