#!/usr/bin/env python
# main.py  –  Mini-RAG API (FastAPI + OpenAI + Pinecone v4)

from __future__ import annotations

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import pinecone

# ────────────────────────────────────────────────────────────────
# 0️⃣  ENV
# ────────────────────────────────────────────────────────────────
load_dotenv()

OPENAI_API_KEY   = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

INDEX_NAME   = os.getenv("PINECONE_INDEX_NAME", "kellogg-rag")
PC_ENV       = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")  # or supply PINECONE_HOST

# ────────────────────────────────────────────────────────────────
# 1️⃣  EMBEDDINGS
# ────────────────────────────────────────────────────────────────
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

# ────────────────────────────────────────────────────────────────
# 2️⃣  PINECONE CLIENT  ➜  VECTOR STORE
# ────────────────────────────────────────────────────────────────
pc    = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PC_ENV)
index = pc.Index(INDEX_NAME)  # pull existing index

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedder,
    text_key="text",          # key we used during ingestion
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ────────────────────────────────────────────────────────────────
# 3️⃣  LLM  +  RAG CHAIN
# ────────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo"),
    temperature=0.3,
)

prompt = ChatPromptTemplate.from_template(
    """You are a concise assistant.

Use ONLY the context below to answer the question.
If the context is not helpful, say you don't know.

<context>
{context}
</context>

Question: {question}
Answer:"""
)

rag_chain = (
    {
        "question": RunnablePassthrough(),
        "context": retriever
        | (lambda docs: "\n\n---\n\n".join(d.page_content for d in docs)),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ────────────────────────────────────────────────────────────────
# 4️⃣  FASTAPI APP
# ────────────────────────────────────────────────────────────────
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
# ── 4️⃣  FASTAPI APP ──────────────────────────────────────────
app = FastAPI(title="Mini-RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or use your frontend URL for tighter security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1️⃣  API routes first
class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat(req: ChatRequest):
    answer = await rag_chain.ainvoke(req.question)
    return {"answer": answer}

# 2️⃣  THEN mount static files
from pathlib import Path
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).parent.parent / "static"
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True))


