import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# Import the existing Self-Healing RAG system
# pyrefly: ignore [missing-import]
from rag import SelfHealingRAG

# Configure logging to match rag.py formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("SelfHealingRAG.API")

# Global reference to the initialized RAG system
rag_instance: SelfHealingRAG | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous context manager to manage application startup and shutdown.
    Ensures that the SelfHealingRAG instance (including database, embeddings,
    and LLM client) is initialized once at startup and reused.
    """
    global rag_instance
    logger.info("Initializing Self-Healing RAG system on startup...")
    start_time = time.time()
    try:
        # Initialize RAG (reuses loaded Chroma vector db and Groq client)
        rag_instance = SelfHealingRAG()
        elapsed_time = time.time() - start_time
        logger.info(f"Self-Healing RAG system initialized successfully in {elapsed_time:.2f}s.")
    except Exception as e:
        logger.critical(f"Critical failure during Self-Healing RAG initialization: {e}", exc_info=True)
        raise e
    
    yield
    
    logger.info("Shutting down Self-Healing RAG API...")


# Create FastAPI application
app = FastAPI(
    title="Self-Healing RAG API",
    description="Production FastAPI wrapper for the Self-Healing RAG Security Assessment Platform.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend/integration support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# --- Request and Response Schemas ---

class QuestionRequest(BaseModel):
    """Schema representing an incoming question query."""
    question: str = Field(..., description="The query question for the RAG system.")

    @field_validator("question", mode="before")
    @classmethod
    def validate_question(cls, v: Any) -> str:
        """Enforces whitespace stripping and rejects empty questions."""
        if not isinstance(v, str):
            raise ValueError("Question must be a string.")
        
        v_stripped = v.strip()
        if len(v_stripped) < 1:
            raise ValueError("Question cannot be empty.")
            
        return v_stripped


class QuestionResponse(BaseModel):
    """Schema representing the RAG query response."""
    question: str = Field(..., description="The original question query.")
    answer: str = Field(..., description="The generated response text from the document context.")
    sources: list[str] = Field(..., description="The file sources from which content was retrieved.")
    retrieved_chunks: int = Field(..., description="The number of retrieved text chunks that met the relevance threshold.")


# --- Exception Handlers ---

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handles request validation errors (e.g., empty or missing questions).
    Returns 400 Bad Request in the exact format required.
    """
    logger.warning(f"Request validation failed for path {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": "Question cannot be empty."}
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catches all unhandled exceptions and processes them as internal server errors.
    Logs the full exception details.
    """
    logger.error(f"Unhandled exception during request processing for {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal RAG processing error."}
    )


# --- Endpoints ---

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint for readiness monitoring and deployment verifications.
    """
    logger.info("Health check endpoint invoked.")
    return {
        "status": "healthy",
        "service": "Self-Healing RAG API"
    }


@app.post("/ask", response_model=QuestionResponse, status_code=status.HTTP_200_OK)
async def ask_question(request: QuestionRequest):
    """
    Main endpoint for asking questions. Integrates with the Self-Healing RAG.
    """
    if rag_instance is None:
        logger.error("RAG system was not initialized properly on startup.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal RAG processing error."
        )

    logger.info(f"Received question: {request.question}")
    start_time = time.time()

    # TODO: Guardrails (future validation of input query)
    
    try:
        # Query the RAG workflow
        # TODO: Self-Healing Retry (future layer for query/retrieval retries)
        # TODO: Critic Agent (future output verification & evaluation)
        result = rag_instance.ask(request.question)
        
        # TODO: Garak Metrics Collection (future security evaluation logging)
        
        processing_time = time.time() - start_time
        logger.info(f"Response generated successfully in {processing_time:.2f}s.")
        
        return QuestionResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result["sources"],
            retrieved_chunks=result["retrieved_chunks"]
        )
        
    except Exception as e:
        logger.error(f"Error occurred during RAG ask flow for query '{request.question}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal RAG processing error."
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
