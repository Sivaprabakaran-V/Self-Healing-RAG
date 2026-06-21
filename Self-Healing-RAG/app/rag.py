import os
import json
import hashlib
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import zipfile
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

# LangChain Imports
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("SelfHealingRAG")


class RateLimitedEmbeddings(Embeddings):
    """
    Custom LangChain Embeddings wrapper that routes embedding calls
    through the SelfHealingRAG instance to enforce Gemini rate limits.
    """
    def __init__(self, google_api_key: str, model_name: str, rag_instance: Any):
        self.underlying = GoogleGenerativeAIEmbeddings(
            google_api_key=google_api_key,
            model=model_name
        )
        self.rag_instance = rag_instance

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Route document embedding calls to the rate-limited method."""
        return self.rag_instance.create_embeddings_with_rate_limit(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed single query with rate limit safety."""
        res = self.rag_instance.create_embeddings_with_rate_limit([text])
        return res[0]


class SelfHealingRAG:
    """
    Self-Healing RAG System featuring registry-based incremental ingestion,
    rate-limited Gemini embeddings, Chroma vector database management,
    MMR-based retrieval, Groq inference, and relevance score verification.
    """
    def __init__(self) -> None:
        logger.info("Initializing Self-Healing RAG system...")
        
        # 1. Load environment variables
        self._load_env()
        
        # 2. Setup directory structure
        self.project_root = Path("E:/AI-Learning/AI Security/Garak/Self-Healing-RAG")
        self.docs_dir = self.project_root / "Documents"
        self.vector_db_dir = self.project_root / "Vector_DB"
        self.registry_path = self.vector_db_dir / "file_registry.json"

        # Auto-create missing directories
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)

        # 3. Initialize embeddings with rate-limiting support
        self.underlying_embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=self.google_api_key,
            model="gemini-embedding-001"
        )
        self.embeddings = RateLimitedEmbeddings(
            google_api_key=self.google_api_key,
            model_name="gemini-embedding-001",
            rag_instance=self
        )

        # 4. Initialize Chroma Vector Database
        self.vectorstore = Chroma(
            collection_name="self_healing_rag",
            embedding_function=self.embeddings,
            persist_directory=str(self.vector_db_dir)
        )

        # 5. Initialize Groq LLM
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=self.groq_api_key,
            temperature=0.0
        )

        # 6. Auto build/update vector store on startup
        self.build_or_update_vectorstore()
        logger.info("Self-Healing RAG initialization complete.")

    def _load_env(self) -> None:
        """Loads and validates configuration from environment variables."""
        # Locate the .env file recursively up to 3 parent directories
        curr_path = Path(__file__).resolve()
        env_path = None
        for parent in [curr_path.parent, curr_path.parent.parent, curr_path.parent.parent.parent]:
            possible_env = parent / ".env"
            if possible_env.exists():
                env_path = possible_env
                break

        if env_path:
            load_dotenv(dotenv_path=env_path)
            logger.info(f"Loaded environment variables from {env_path}")
        else:
            load_dotenv()
            logger.warning("Could not find local .env file. Relying on system environment variables.")

        # Required variables
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")

        if not self.google_api_key:
            raise ValueError("Missing required environment variable: GOOGLE_API_KEY")
        if not self.groq_api_key:
            raise ValueError("Missing required environment variable: GROQ_API_KEY")

        # Optional variables with defaults
        self.embedding_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", 20))
        self.embedding_batch_delay = float(os.getenv("EMBEDDING_BATCH_DELAY", 1.0))
        self.min_relevance_score = float(os.getenv("MIN_RELEVANCE_SCORE", 0.70))

        logger.info(
            f"Config - Batch Size: {self.embedding_batch_size}, "
            f"Batch Delay: {self.embedding_batch_delay}s, "
            f"Min Relevance Threshold: {self.min_relevance_score}"
        )

    def calculate_file_hash(self, file_path: str) -> str:
        """Calculates SHA256 hash of a file for change and duplicate detection."""
        sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256.update(byte_block)
            return sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            raise e

    def load_registry(self) -> dict[str, dict[str, str]]:
        """Loads the file registry from storage."""
        if not self.registry_path.exists():
            logger.info("Registry file does not exist. Initializing empty registry.")
            return {}
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            registry = {}
            if isinstance(data, list):
                for entry in data:
                    name = entry.get("file_name")
                    if name:
                        registry[name] = {
                            "file_hash": entry.get("file_hash"),
                            "last_modified": entry.get("last_modified")
                        }
            elif isinstance(data, dict):
                for name, entry in data.items():
                    registry[name] = {
                        "file_hash": entry.get("file_hash"),
                        "last_modified": entry.get("last_modified")
                    }
            logger.info(f"Loaded {len(registry)} files from registry.")
            return registry
        except Exception as e:
            logger.error(f"Error loading registry: {e}")
            return {}

    def save_registry(self, registry: dict[str, dict[str, str]]) -> None:
        """Saves the file registry matching the requested schema exactly."""
        try:
            list_data = []
            for name, entry in registry.items():
                list_data.append({
                    "file_name": name,
                    "file_hash": entry["file_hash"],
                    "last_modified": entry["last_modified"]
                })
            
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(list_data, f, indent=2)
            logger.info("File registry saved successfully.")
        except Exception as e:
            logger.error(f"Error saving registry: {e}")

    def scan_documents(self) -> tuple[list[str], list[str], dict[str, dict[str, str]]]:
        """
        Scans the Documents directory recursively.
        Detects new, modified, and deleted files.
        Checks for duplicate files (identical hashes).
        Returns:
          - files_to_process: list of absolute file paths to ingest.
          - files_to_delete: list of file names (relative paths) to remove.
          - current_scanned: dict of file_name (relative path) to metadata (hash, modified time).
        """
        registry = self.load_registry()
        current_scanned = {}
        supported_exts = {".pdf", ".docx", ".txt", ".md"}
        
        logger.info(f"Scanning Documents directory recursively: {self.docs_dir}")
        for root, _, files in os.walk(self.docs_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext not in supported_exts:
                    continue
                
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, self.docs_dir).replace("\\", "/")
                
                try:
                    file_hash = self.calculate_file_hash(abs_path)
                    mtime = os.path.getmtime(abs_path)
                    last_modified = datetime.fromtimestamp(mtime).isoformat()
                    
                    current_scanned[rel_path] = {
                        "abs_path": abs_path,
                        "file_hash": file_hash,
                        "last_modified": last_modified
                    }
                except Exception as e:
                    logger.error(f"Failed to scan file {abs_path}: {e}")

        # Detect deleted files
        files_to_delete = []
        for reg_file in registry:
            if reg_file not in current_scanned:
                logger.info(f"Detected deleted file: {reg_file}")
                files_to_delete.append(reg_file)

        # Detect duplicate files & find files to process
        processed_hashes = set()
        # Seed processed_hashes with files currently in registry that are unchanged
        for f_name, entry in registry.items():
            if f_name in current_scanned:
                cur_hash = current_scanned[f_name]["file_hash"]
                cur_mod = current_scanned[f_name]["last_modified"]
                if cur_hash == entry["file_hash"] and cur_mod == entry["last_modified"]:
                    processed_hashes.add(entry["file_hash"])

        files_to_process = []
        for rel_path, info in current_scanned.items():
            f_hash = info["file_hash"]
            f_mod = info["last_modified"]
            abs_path = info["abs_path"]

            # Handle identical hash already processed
            if f_hash in processed_hashes:
                if rel_path in registry and registry[rel_path]["file_hash"] == f_hash:
                    # Same file, unchanged.
                    continue
                else:
                    logger.info("Duplicate document detected")
                    logger.info("Skipping embedding")
                    continue

            if rel_path not in registry:
                logger.info(f"New file detected: {rel_path}")
                files_to_process.append(abs_path)
                processed_hashes.add(f_hash)
            elif registry[rel_path]["file_hash"] != f_hash or registry[rel_path]["last_modified"] != f_mod:
                logger.info(f"Modified file detected: {rel_path}")
                files_to_process.append(abs_path)
                processed_hashes.add(f_hash)
                files_to_delete.append(rel_path)  # Re-index requires deleting old chunks first

        return files_to_process, files_to_delete, current_scanned

    def _read_docx(self, file_path: str) -> str:
        """Reads text from a DOCX file natively and dependency-free."""
        try:
            with zipfile.ZipFile(file_path) as docx:
                tree = ET.fromstring(docx.read("word/document.xml"))
            
            namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            paragraphs = tree.findall(".//w:p", namespaces)
            text_list = []
            
            for paragraph in paragraphs:
                texts = paragraph.findall(".//w:t", namespaces)
                if texts:
                    text_list.append("".join(t.text for t in texts if t.text))
            
            return "\n".join(text_list)
        except Exception as e:
            logger.error(f"Error parsing DOCX file {file_path}: {e}")
            raise e

    def _read_file_content(self, file_path: str) -> list[Document]:
        """Helper to read supported file contents into Document objects."""
        ext = os.path.splitext(file_path)[1].lower()
        documents = []
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
                documents = loader.load()
            elif ext == ".docx":
                text = self._read_docx(file_path)
                documents = [Document(page_content=text, metadata={"source": file_path})]
            elif ext in (".txt", ".md"):
                loader = TextLoader(file_path, encoding="utf-8")
                documents = loader.load()
            else:
                logger.warning(f"Unsupported file type: {ext} for file {file_path}")
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")

        # Ensure source path is correct in metadata
        for doc in documents:
            if "source" not in doc.metadata:
                doc.metadata["source"] = file_path
        return documents

    def process_new_documents(self, file_paths: list[str]) -> list[Document]:
        """Loads and chunks documents, merging metadata and constructing chunk IDs."""
        all_chunks = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        for file_path in file_paths:
            logger.info(f"Loaded file: {file_path}")
            docs = self._read_file_content(file_path)
            if not docs:
                logger.warning(f"No content loaded from {file_path}")
                continue

            file_hash = self.calculate_file_hash(file_path)
            ext = os.path.splitext(file_path)[1].lower().lstrip(".")
            rel_name = os.path.relpath(file_path, self.docs_dir).replace("\\", "/")

            chunks = splitter.split_documents(docs)
            logger.info(f"Chunk creation: Split {rel_name} into {len(chunks)} chunks.")

            for idx, chunk in enumerate(chunks):
                chunk_num = idx + 1
                chunk_id = f"{file_hash}_{chunk_num:03d}"

                new_metadata = chunk.metadata.copy()
                new_metadata.update({
                    "source_file": os.path.basename(file_path),
                    "chunk_id": chunk_id,
                    "document_type": ext,
                    "file_hash": file_hash
                })
                chunk.metadata = new_metadata
                all_chunks.append(chunk)

        return all_chunks

    def create_embeddings_with_rate_limit(self, texts: list[str]) -> list[list[float]]:
        """
        Embeds list of texts using the underlying Gemini embedding model with rate limiting.
        Handles batching, delay between batches, and retries on HTTP 429 using exponential backoff.
        """
        batch_size = self.embedding_batch_size
        batch_delay = self.embedding_batch_delay
        all_embeddings = []
        total_texts = len(texts)
        num_batches = (total_texts + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, total_texts)
            batch_texts = texts[start_idx:end_idx]

            logger.info(f"Processing batch {i + 1} of {num_batches}")
            batch_embeddings = self._embed_batch_with_retry(batch_texts)
            all_embeddings.extend(batch_embeddings)

            if i < num_batches - 1:
                logger.info(f"Embedding creation: Sleeping {batch_delay}s between batches.")
                time.sleep(batch_delay)

        return all_embeddings

    def _embed_batch_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Embeds a single batch with exponential backoff on HTTP 429."""
        backoff = [1, 2, 4, 8, 16]
        max_retries = 5

        for attempt in range(max_retries + 1):
            try:
                return self.underlying_embeddings.embed_documents(texts)
            except Exception as e:
                err_msg = str(e)
                is_rate_limit = (
                    "429" in err_msg 
                    or "ResourceExhausted" in err_msg 
                    or "rate limit" in err_msg.lower() 
                    or "quota" in err_msg.lower()
                )

                if is_rate_limit and attempt < max_retries:
                    delay = backoff[attempt]
                    logger.warning(f"Rate limit encountered: {err_msg}")
                    logger.info(f"Retrying in {delay} seconds")
                    time.sleep(delay)
                else:
                    logger.error(f"Failed to embed batch: {err_msg}")
                    raise e
        raise RuntimeError("Failed to embed batch after max retries.")

    def build_or_update_vectorstore(self) -> None:
        """Orchestrates indexing pipeline. Removes old files and embeds new/modified documents."""
        logger.info("Chroma updates: Checking vector store status...")
        registry = self.load_registry()
        to_process, to_delete, current_scanned = self.scan_documents()

        # 1. Clean up old indexes
        for rel_path in to_delete:
            if rel_path in registry:
                old_hash = registry[rel_path]["file_hash"]
                logger.info(f"Deleting old Chroma chunks for: {rel_path}")
                try:
                    res = self.vectorstore.get(where={"file_hash": old_hash})
                    if res and "ids" in res and res["ids"]:
                        self.vectorstore.delete(ids=res["ids"])
                        logger.info(f"Chroma updates: Deleted {len(res['ids'])} old chunks.")
                except Exception as e:
                    logger.error(f"Failed to delete chunks for {rel_path}: {e}")
                registry.pop(rel_path, None)

        # 2. Ingest and index new/modified files
        if to_process:
            chunks = self.process_new_documents(to_process)
            if chunks:
                chunk_ids = [c.metadata["chunk_id"] for c in chunks]
                logger.info(f"Chroma updates: Adding {len(chunks)} chunks to collection.")
                try:
                    self.vectorstore.add_documents(documents=chunks, ids=chunk_ids)
                except Exception as e:
                    logger.error(f"Failed to add chunks to Chroma: {e}")
                    raise e

                # Update registry metadata
                for file_path in to_process:
                    rel_path = os.path.relpath(file_path, self.docs_dir).replace("\\", "/")
                    if rel_path in current_scanned:
                        registry[rel_path] = {
                            "file_hash": current_scanned[rel_path]["file_hash"],
                            "last_modified": current_scanned[rel_path]["last_modified"]
                        }
            else:
                logger.info("No chunks generated from process list.")
        else:
            logger.info("No new/modified documents to index.")

        if hasattr(self.vectorstore, "persist"):
            self.vectorstore.persist()

        self.save_registry(registry)

    def _get_relevance_scores(self, query: str, docs: list[Document]) -> list[float]:
        """Calculates cosine similarity between query and retrieved document embeddings."""
        if not docs:
            return []

        query_vector = self.embeddings.embed_query(query)
        query_arr = np.array(query_vector)

        chunk_ids = [doc.metadata.get("chunk_id") for doc in docs if doc.metadata.get("chunk_id")]
        embeddings_dict = {}

        if len(chunk_ids) == len(docs):
            try:
                res = self.vectorstore.get(ids=chunk_ids, include=["embeddings"])
                if res and "embeddings" in res and res["embeddings"] is not None and len(res["embeddings"]) > 0:
                    for cid, emb in zip(res["ids"], res["embeddings"]):
                        embeddings_dict[cid] = emb
            except Exception as e:
                logger.warning(f"Failed to retrieve embeddings from Chroma: {e}. Falling back to dynamic embedding.")

        scores = []
        for doc in docs:
            cid = doc.metadata.get("chunk_id")
            emb = None
            if cid in embeddings_dict:
                emb = embeddings_dict[cid]
            else:
                try:
                    # Dynamically calculate if missing from DB response
                    emb = self.underlying_embeddings.embed_documents([doc.page_content])[0]
                except Exception as e:
                    logger.error(f"Failed to embed page content: {e}")

            if emb is not None:
                doc_arr = np.array(emb)
                dot = np.dot(query_arr, doc_arr)
                norm_q = np.linalg.norm(query_arr)
                norm_d = np.linalg.norm(doc_arr)
                score = float(dot / (norm_q * norm_d)) if norm_q > 0 and norm_d > 0 else 0.0
            else:
                score = 0.0
            scores.append(score)

        return scores

    def retrieve_context(self, question: str) -> dict[str, Any]:
        """
        Retrieves context documents for a question using MMR.
        Calculates relevance scores for the retrieved documents.
        """
        logger.info(f"Retrieval operations: Retrieving context for question: '{question}'")
        try:
            docs = self.vectorstore.max_marginal_relevance_search(
                query=question,
                k=5,
                fetch_k=20
            )
            logger.info(f"Retrieval operations: Retracted {len(docs)} documents via MMR.")
            
            scores = self._get_relevance_scores(question, docs)
            metadata_list = [doc.metadata for doc in docs]
            sources = list(set(doc.metadata.get("source_file", "") for doc in docs if doc.metadata.get("source_file")))

            return {
                "documents": docs,
                "metadata": metadata_list,
                "sources": sources,
                "relevance_scores": scores
            }
        except Exception as e:
            logger.error(f"Error during context retrieval: {e}")
            raise e

    def generate_answer(self, question: str, context: str) -> str:
        """
        Generates an answer from the query and context using the Groq LLM.
        Applies strict system prompts and instructions.
        """
        logger.info("LLM generation: Invoking Groq LLM...")
        system_prompt = (
            "You are an enterprise document assistant.\n\n"
            "Rules:\n"
            "1. Answer ONLY from the supplied document context.\n"
            "2. Never use outside knowledge.\n"
            "3. Never hallucinate.\n"
            "4. Never make assumptions.\n"
            "5. If information is not found in the supplied documents, respond exactly:\n"
            "\"I could not find sufficient information in the uploaded documents.\"\n"
            "6. Cite source filenames used for the answer.\n"
            "7. If context is weak or uncertain, return the exact fallback response above."
        )

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ]
            response = self.llm.invoke(messages)
            answer = response.content.strip()
            logger.info("LLM generation: Response generated successfully.")
            return answer
        except Exception as e:
            logger.error(f"Error generating answer from LLM: {e}")
            raise e

    def evaluate_response(self) -> None:
        """
        Placeholder for response evaluation.
        # TODO: Integrate Critic Agent
        # TODO: Integrate Hallucination Detection
        """
        pass

    def verify_retrieval_quality(self) -> None:
        """
        Placeholder for retrieval quality verification.
        # TODO: Integrate Retrieval Verification
        # TODO: Integrate Self-Healing Retry Logic
        """
        pass

    def ask(self, question: str) -> dict[str, Any]:
        """
        Executes the full RAG query flow.
        Retrieves context, validates relevance, generates answer, and formats response.
        """
        logger.info(f"Startup: Querying: '{question}'")
        
        # 1. Retrieve context
        retrieval_res = self.retrieve_context(question)
        docs = retrieval_res["documents"]
        scores = retrieval_res["relevance_scores"]

        # 2. Filter relevance scores
        filtered_docs = []
        filtered_sources = set()

        for doc, score in zip(docs, scores):
            if score >= self.min_relevance_score:
                filtered_docs.append(doc)
                source_file = doc.metadata.get("source_file")
                if source_file:
                    filtered_sources.add(source_file)
            else:
                logger.info(
                    f"Filtered out chunk {doc.metadata.get('chunk_id')} "
                    f"with relevance score {score:.4f} (below threshold {self.min_relevance_score})"
                )

        fallback_answer = "I could not find sufficient information in the uploaded documents."

        # Verify answer validation triggers
        if not docs or not filtered_docs:
            logger.warning("No documents retrieved or all chunks are below the relevance threshold.")
            return {
                "question": question,
                "answer": fallback_answer,
                "sources": [],
                "retrieved_chunks": 0
            }

        context_text = "\n\n".join(doc.page_content for doc in filtered_docs)
        if not context_text.strip():
            logger.warning("Retrieved context content is empty.")
            return {
                "question": question,
                "answer": fallback_answer,
                "sources": [],
                "retrieved_chunks": 0
            }

        # 3. Generate response using LLM
        try:
            answer = self.generate_answer(question, context_text)
            return {
                "question": question,
                "answer": answer,
                "sources": sorted(list(filtered_sources)),
                "retrieved_chunks": len(filtered_docs)
            }
        except Exception as e:
            logger.error(f"Error handling LLM invocation: {e}")
            return {
                "question": question,
                "answer": fallback_answer,
                "sources": [],
                "retrieved_chunks": 0
            }


if __name__ == "__main__":
    rag = SelfHealingRAG()

    while True:
        query = input("\nAsk a question (type exit to quit): ")

        if query.lower() == "exit":
            break

        result = rag.ask(query)

        print("\n" + "="*50)
        print("Answer:")
        print(result["answer"])
        print("\nSources:")
        print(result["sources"])

        print("\nRetrieved Chunks:")
        print(result["retrieved_chunks"])
        print("="*50)
