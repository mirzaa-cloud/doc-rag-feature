from langchain_huggingface import HuggingFaceEmbeddings


class GroqEmbeddingService:
    """
    HuggingFace embeddings service - no API quota limits
    Runs locally for fast, unlimited embeddings
    """
    
    def __init__(self):
        # Using a well-known, fast, and local embedding model
        self.embed_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

    def embed_texts(self, texts):
        """Batch embed documents"""
        return self.embed_model.embed_documents(texts)

    def embed_query(self, text):
        """Embed a single query"""
        return self.embed_model.embed_query(text)
