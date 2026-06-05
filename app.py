"""
RAG System with BitNet b1.58 inference
Architecture: PyMuPDF4LLM → LangChain Splitter → ChromaDB → BitNet via subprocess/OpenAI-compatible API
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import List, Optional

import requests
from dotenv import load_dotenv

# # LangChain
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.schema import Document

# Por ESTAS (versão 1.x correta):
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings  
from langchain_core.documents import Document

# PyMuPDF4LLM — extração de PDF em Markdown (suporta tabelas e colunas)
import pymupdf4llm

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Variáveis de ambiente com fallback
PDF_FOLDER       = Path(os.getenv("PDF_FOLDER", "./documents"))
CHROMA_DB_PATH   = Path(os.getenv("CHROMA_DB_PATH", "./chroma_db"))
EMBED_MODEL      = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")   # leve, ~80 MB
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP", "200"))          # overlap vital para contexto
TOP_K            = int(os.getenv("TOP_K", "4"))                    # chunks recuperados por query

# Endpoint do bitnet.cpp (OpenAI-compatible via llama-server embutido)
BITNET_ENDPOINT  = os.getenv("BITNET_ENDPOINT", "http://localhost:11434/v1/chat/completions")
BITNET_MODEL     = os.getenv("BITNET_MODEL", "bitnet-b1.58-2B-4T")


# ---------------------------------------------------------------------------
# MÓDULO 1: Ingestão de PDFs
# ---------------------------------------------------------------------------
import fitz  # PyMuPDF

def load_pdfs(folder: Path) -> List[Document]:
    """
    Lê PDFs usando PyMuPDF puro (fitz) — mais estável que pymupdf4llm.
    """
    if not folder.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {folder}")

    pdf_files = list(folder.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"Nenhum arquivo PDF encontrado em: {folder}")

    logger.info(f"PDFs encontrados: {[f.name for f in pdf_files]}")

    documents: List[Document] = []

    for pdf_path in pdf_files:
        try:
            doc = fitz.open(str(pdf_path))
            text_parts = []
            
            for page in doc:
                # Extrai texto da página (pode adicionar flags para melhor layout se necessário)
                text_parts.append(page.get_text("text"))
            
            doc.close()
            md_text = "\n".join(text_parts)

            if not md_text.strip():
                logger.warning(f"PDF vazio ou ilegível: {pdf_path.name}")
                continue

            documents.append(Document(
                page_content=md_text,
                metadata={"source": pdf_path.name, "path": str(pdf_path)}
            ))
            logger.info(f"Carregado: {pdf_path.name} ({len(md_text)} chars)")

        except Exception as e:
            logger.error(f"Erro ao ler {pdf_path.name}: {e}")

    if not documents:
        raise RuntimeError("Nenhum documento foi carregado com sucesso.")

    return documents
# def load_pdfs(folder: Path) -> List[Document]:
#     """
#     Lê todos os PDFs em `folder`, converte para Markdown usando pymupdf4llm
#     (preserva tabelas, listas e estrutura), e retorna uma lista de Documents.
#     """
#     if not folder.exists():
#         raise FileNotFoundError(f"Pasta não encontrada: {folder}")

#     pdf_files = list(folder.glob("*.pdf"))
#     if not pdf_files:
#         raise ValueError(f"Nenhum arquivo PDF encontrado em: {folder}")

#     logger.info(f"PDFs encontrados: {[f.name for f in pdf_files]}")

#     documents: List[Document] = []

#     for pdf_path in pdf_files:
#         try:
#             # pymupdf4llm extrai texto estruturado em Markdown — superior ao
#             # PyPDFLoader simples para PDFs com tabelas, colunas e imagens captioned
#             md_text = pymupdf4llm.to_markdown(str(pdf_path))

#             if not md_text.strip():
#                 logger.warning(f"PDF vazio ou ilegível: {pdf_path.name}")
#                 continue

#             documents.append(Document(
#                 page_content=md_text,
#                 metadata={"source": pdf_path.name, "path": str(pdf_path)}
#             ))
#             logger.info(f"Carregado: {pdf_path.name} ({len(md_text)} chars)")

#         except Exception as e:
#             logger.error(f"Erro ao ler {pdf_path.name}: {e}")

#     if not documents:
#         raise RuntimeError("Nenhum documento foi carregado com sucesso.")

#     return documents


# ---------------------------------------------------------------------------
# MÓDULO 2: Chunking e Indexação no ChromaDB
# ---------------------------------------------------------------------------

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Divide os documentos em chunks de tamanho `CHUNK_SIZE` com `CHUNK_OVERLAP`
    de sobreposição. O overlap garante que conceitos que cruzam a fronteira entre
    chunks não sejam perdidos na recuperação.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # respeita parágrafos e frases
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Total de chunks gerados: {len(chunks)}")
    return chunks


def build_vectorstore(chunks: List[Document]) -> Chroma:
    """
    Gera embeddings locais com SentenceTransformers e persiste no ChromaDB.
    `all-MiniLM-L6-v2` é leve (~80 MB) e roda 100% offline em CPU.
    
    O ChromaDB armazena os vetores em disco — na próxima execução, basta
    chamar `load_vectorstore()` sem reindexar.
    """
    logger.info(f"Gerando embeddings com modelo: {EMBED_MODEL}")
    embedding_fn = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        persist_directory=str(CHROMA_DB_PATH),
    )
    logger.info(f"Vectorstore salvo em: {CHROMA_DB_PATH}")
    return vectorstore


def load_vectorstore() -> Chroma:
    """Carrega um ChromaDB já existente no disco sem reindexar."""
    embedding_fn = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        persist_directory=str(CHROMA_DB_PATH),
        embedding_function=embedding_fn,
    )


# ---------------------------------------------------------------------------
# MÓDULO 3: Recuperação de Contexto (Retrieval)
# ---------------------------------------------------------------------------

def retrieve_context(vectorstore: Chroma, query: str) -> List[Document]:
    """
    Realiza busca por similaridade semântica. Os TOP_K chunks mais relevantes
    são retornados — eles formarão o contexto injetado no prompt do BitNet.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
    results = retriever.invoke(query)
    logger.info(f"Recuperados {len(results)} chunks para a query.")
    return results


def format_context(chunks: List[Document]) -> str:
    """
    Monta um bloco de contexto formatado a partir dos chunks recuperados.
    Cada chunk recebe sua fonte como cabeçalho para rastreabilidade.
    """
    parts = []
    for i, doc in enumerate(chunks, 1):
        source = doc.metadata.get("source", "desconhecido")
        parts.append(f"[Trecho {i} — {source}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# MÓDULO 4: Geração com BitNet (via OpenAI-compatible API)
# ---------------------------------------------------------------------------

def generate_answer(context: str, question: str) -> str:
    """
    Envia o contexto recuperado + pergunta ao BitNet b1.58 rodando localmente.
    
    O bitnet.cpp expõe uma API compatível com OpenAI (endpoint /v1/chat/completions)
    quando iniciado com `python run_inference.py --server`. Isso permite que qualquer
    cliente HTTP padrão interaja com o modelo sem binding Python específico.
    
    O prompt segue o padrão RAG:
      1. Instrução de sistema: define o comportamento (responder baseado no contexto)
      2. Contexto: os chunks recuperados são injetados aqui
      3. Pergunta do usuário
    """
    system_prompt = (
        "Você é um assistente especializado em responder perguntas com base "
        "exclusivamente nos trechos de documentos fornecidos. "
        "Se a resposta não estiver nos trechos, diga claramente que não encontrou "
        "a informação nos documentos. Seja preciso e cite o trecho relevante."
    )

    user_message = (
        f"Contexto dos documentos:\n\n{context}\n\n"
        f"Pergunta: {question}\n\n"
        "Responda com base apenas no contexto acima."
    )

    payload = {
        "model": BITNET_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "temperature": 0.2,   # baixo para respostas determinísticas e factuais
        "max_tokens": 1024,
        "stream": False,
    }

    try:
        response = requests.post(
            BITNET_ENDPOINT,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    except requests.exceptions.ConnectionError:
        return (
            "ERRO: Não foi possível conectar ao BitNet. "
            f"Verifique se o servidor está rodando em {BITNET_ENDPOINT}."
        )
    except requests.exceptions.HTTPError as e:
        return f"ERRO HTTP: {e}"
    except KeyError:
        return f"ERRO: Resposta inesperada da API: {response.text}"


# ---------------------------------------------------------------------------
# PIPELINE PRINCIPAL
# ---------------------------------------------------------------------------

def ingest_pipeline():
    """Executa a ingestão completa: PDF → chunks → ChromaDB."""
    logger.info("=== INICIANDO INGESTÃO ===")
    documents = load_pdfs(PDF_FOLDER)
    chunks    = split_documents(documents)
    build_vectorstore(chunks)
    logger.info("=== INGESTÃO CONCLUÍDA ===")


def query_pipeline(question: str) -> str:
    """Executa o pipeline de consulta: query → retrieval → geração."""
    if not CHROMA_DB_PATH.exists():
        raise RuntimeError(
            "Vectorstore não encontrado. Execute primeiro: python app.py ingest"
        )

    vectorstore = load_vectorstore()
    chunks      = retrieve_context(vectorstore, question)
    context     = format_context(chunks)
    answer      = generate_answer(context, question)
    return answer


def main():
    if len(sys.argv) < 2:
        print("Uso: python app.py [ingest | query 'sua pergunta']")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "ingest":
        ingest_pipeline()

    elif command == "query":
        if len(sys.argv) < 3:
            print("Forneça a pergunta: python app.py query 'sua pergunta aqui'")
            sys.exit(1)
        question = " ".join(sys.argv[2:])
        print(f"\nPergunta: {question}\n")
        answer = query_pipeline(question)
        print(f"Resposta:\n{answer}\n")

    else:
        print(f"Comando desconhecido: {command}")
        print("Comandos válidos: ingest | query")
        sys.exit(1)


if __name__ == "__main__":
    main()