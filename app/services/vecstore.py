from langchain_qdrant import QdrantVectorStore
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PayloadSchemaType
from app.config import settings
from app.services.groq_embeddings import GroqEmbeddingService

class GroqEmbeddingsAdapter(Embeddings):
    def __init__(self, groq_service: GroqEmbeddingService = None):
        self.groq = groq_service or GroqEmbeddingService()

    def embed_documents(self, texts):
        return self.groq.embed_texts(texts)

    def embed_query(self, text):
        return self.groq.embed_query(text)

def get_qdrant_vectorstore(collection_name: str, adapter: GroqEmbeddingsAdapter = None):
    adapter = adapter or GroqEmbeddingsAdapter()
    client = QdrantClient(
        url=settings.QDRANT_HOST,
        api_key=settings.QDRANT_API_KEY,
        timeout=30,
    )
    existing_collections = [c.name for c in client.get_collections().collections or []]
    if collection_name not in existing_collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=settings.EMBEDDING_DIM,
                distance=models.Distance.COSINE,
            ),
        )
        print(f"Created new collection: {collection_name}")
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name="source",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            print(f"Created payload index for 'source' on {collection_name}")
        except Exception as e:
            print(f"Failed to create payload index for '{collection_name}': {e}")
    else:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name="source",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            print(f"Ensured payload index for 'source' on {collection_name}")
        except Exception as e:
            print(f"Payload index already exists or cannot be recreated: {e}")
    vectstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=adapter,
    )
    return vectstore, client

def delete_documents_by_source(session_id: str, filename: str):
    client = QdrantClient(
        url=settings.QDRANT_HOST,
        api_key=settings.QDRANT_API_KEY,
        timeout=30,
    )
    points, _ = client.scroll(
        collection_name=session_id,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.source",
                    match=models.MatchValue(
                        value=filename
                    ),
                )
            ]
        ),
        with_payload=True,
        limit=10000
    )
    if not points:
        print(f"No vectors found for '{filename}' in {session_id}.")
        return
    ids = [p.id for p in points]
    client.delete(
        collection_name=session_id,
        points_selector=models.PointIdsList(points=ids)
    )
    print(f"Deleted {len(ids)} vectors for '{filename}' in session '{session_id}'.")

def create_qdrant_collection(collection_name: str, embedding_dim: int = None):
    client = QdrantClient(
        url=settings.QDRANT_HOST,
        api_key=settings.QDRANT_API_KEY,
        timeout=30,
    )
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=embedding_dim or settings.EMBEDDING_DIM,
                distance=models.Distance.COSINE,
            ),
        )
        print(f"Created new collection: {collection_name}")
        client.create_payload_index(
            collection_name=collection_name,
            field_name="source",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        print(f"Created payload index for 'source' on {collection_name}")
    except Exception as e:
        print(f"Error creating Qdrant collection/index: {e}")
    return client
