#!/usr/bin/env python
"""
Embed documents into a serverless Pinecone index.

Examples
--------
# Ingest whole folder (defaults to ./data)
python ingest.py --folder data/

# Ingest a single file
python ingest.py --file docs/Guide.pdf
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import typer
from dotenv import load_dotenv
from tqdm import tqdm
from unstructured.partition.auto import partition

# ── LangChain / Pinecone ---------------------------------------------------
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Optional PDF fallback (only used if import succeeds)
try:
    from PyPDF2 import PdfReader  # noqa: WPS433
except ImportError:  # pragma: no cover
    PdfReader = None  # type: ignore

load_dotenv()
CLI = typer.Typer(help="Chunk, embed & push docs into Pinecone")


# ── Helpers ----------------------------------------------------------------
def load_file(path: Path) -> Optional[str]:
    """Return raw text or None if nothing extractable."""
    # ① Try `unstructured`
    try:
        elems = partition(filename=str(path))
        txt = "\n".join(e.text for e in elems if e.text and e.text.strip())
        if txt.strip():
            return txt
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"[unstructured ⤫] {path.name}: {exc}")

    # ② Plain-text fallback
    if path.suffix.lower() == ".txt":
        try:
            return path.read_text(encoding="utf-8", errors="ignore").strip() or None
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"[txt ⤫] {path.name}: {exc}")

    # ③ PDF fallback
    if path.suffix.lower() == ".pdf" and PdfReader:
        try:
            reader = PdfReader(str(path))
            txt = "\n".join(p.extract_text() or "" for p in reader.pages).strip()
            return txt or None
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"[pypdf ⤫] {path.name}: {exc}")

    return None


def chunk(text: str, size: int = 800, overlap: int = 100) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)


def get_vectorstore(embedder: OpenAIEmbeddings) -> PineconeVectorStore:
    api_key = os.environ["PINECONE_API_KEY"]
    host = os.getenv("PINECONE_HOST")
    index_name = os.getenv("PINECONE_INDEX_NAME", "kellogg-rag")
    region = os.getenv("PINECONE_REGION", "us-east-1")
    dims = 1536  # text-embedding-3-small

    pc = Pinecone(api_key=api_key)

    if host:
        index = pc.Index(host=host)
    else:
        if index_name not in [idx["name"] for idx in pc.list_indexes()]:
            typer.echo(f"[+] Creating Pinecone index '{index_name}' …")
            pc.create_index(
                name=index_name,
                dimension=dims,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=region),
            )
        index = pc.Index(index_name)

    return PineconeVectorStore(index, embedder, text_key="text")


# ── CLI command ------------------------------------------------------------
@CLI.command()
def run(
    folder: Path = typer.Option(None, help="Folder with docs"),
    file: Path = typer.Option(None, help="Single file to ingest"),
) -> None:
    """Embed & upsert documents into Pinecone."""
    # Validate input flags
    if (folder is None and file is None) or (folder and file):
        typer.secho("⚠️  Specify *either* --folder or --file", fg="red", err=True)
        raise typer.Exit(1)

    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = get_vectorstore(embedder)

    paths = [file] if file else list(folder.glob("**/*"))
    docs, metas = [], []

    for fp in tqdm(paths, desc="Parsing", colour="#38bdf8"):
        if not fp.is_file():
            continue
        text = load_file(fp)
        tqdm.write(f"{fp.name:<35} ➜ {'✔ text' if text else '∅ empty'}")
        if not text:
            continue
        for i, ck in enumerate(chunk(text)):
            docs.append(ck)
            metas.append({"source": str(fp), "chunk": i})

    if not docs:
        typer.secho("Nothing to embed 🤷", fg="yellow")
        raise typer.Exit(1)

    typer.echo(f"📄 {len(docs):,} chunks – embedding & upserting …")
    vectorstore.add_texts(docs, metadatas=metas)
    typer.secho("✅ Ingestion complete", fg="green")


if __name__ == "__main__":
    CLI()
