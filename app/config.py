from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Supabase
    SUPABASE_URL: str
    SUPABASE_KEY: str

    # Groq
    GROQ_API_KEY: str
    GROQ_CHAT_ENDPOINT: str = "https://api.groq.com/openai/v1/chat/completions"
    GROQ_EMBED_ENDPOINT: str = "https://api.groq.com/openai/v1/embeddings"

    # Models
    GROQ_GUARD_MODEL: str = "meta-llama/llama-guard-4-12b"
    GROQ_RAG_MODEL: str ="llama-3.3-70b-versatile"
    

    # Qdrant cloud vector DB config
    QDRANT_API_KEY: str  
    QDRANT_HOST: str = "https://c7fde2a2-fd38-4648-897e-f1e66a5a01d8.us-east4-0.gcp.cloud.qdrant.io"

    # Embedding model dimension (HuggingFace MiniLM = 384)
    EMBEDDING_DIM: int = 768

    # Chunking
    CHUNK_SIZE: int = 300
    CHUNK_OVERLAP: int = 60
    TOP_K: int = 15

    # File limits
    MAX_FILE_SIZE_MB: int = 20
    ALLOWED_EXT: tuple = (".pdf", ".docx", ".txt")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
