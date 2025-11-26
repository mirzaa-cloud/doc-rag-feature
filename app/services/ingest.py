import time
import logging
from typing import Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import models
from qdrant_client.http.models import PayloadSchemaType
from app.services.groq_embeddings import GroqEmbeddingService
from app.services.vecstore import GroqEmbeddingsAdapter, get_qdrant_vectorstore
from app.config import settings

logger = logging.getLogger(__name__)

def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.CHUNK_SIZE,
        chunk_overlap=overlap or settings.CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(text)
    logger.debug(f"Split text into {len(chunks)} chunks (size={chunk_size or settings.CHUNK_SIZE}, overlap={overlap or settings.CHUNK_OVERLAP})")
    return chunks

def ingest_documents(session_id: str, docs: Dict[str, str]):
    logger.info(f"{'='*80}")
    logger.info(f"[INGESTION] Starting ingestion for session: {session_id}")
    logger.info(f"[INGESTION] Documents to ingest: {list(docs.keys())}")
    logger.info(f"{'='*80}")

    groq_service = GroqEmbeddingService()
    adapter = GroqEmbeddingsAdapter(groq_service)
    vect, client = get_qdrant_vectorstore(collection_name=session_id, adapter=adapter)
    logger.info(f"[INGESTION] Vector store initialized for collection: {session_id}")

    try:
        logger.info(f"[INGESTION] Creating payload index for 'metadata.source' field...")
        client.create_payload_index(
            collection_name=session_id,
            field_name="metadata.source",
            field_schema=PayloadSchemaType.KEYWORD
        )
        logger.info(f"[INGESTION] ✅ Payload index created successfully for 'metadata.source'")
        time.sleep(0.5)
        collection_info = client.get_collection(collection_name=session_id)
        if collection_info.payload_schema:
            if 'metadata.source' in collection_info.payload_schema or 'metadata' in collection_info.payload_schema:
                logger.info(f"[INGESTION] ✅ Verified: 'metadata.source' field is indexed")
                logger.debug(f"[INGESTION] Payload schema: {collection_info.payload_schema}")
            else:
                logger.warning(f"[INGESTION] ⚠️ 'metadata.source' not found in payload schema")
                logger.warning(f"[INGESTION] Available fields: {list(collection_info.payload_schema.keys())}")
        else:
            logger.warning(f"[INGESTION] ⚠️ No payload schema found after index creation")
    except Exception as e:
        logger.error(f"[INGESTION] ❌ Failed to create payload index: {e}")
        logger.error(f"[INGESTION] ⚠️ File filtering will NOT work without payload index!")
        logger.error(f"[INGESTION] This is a critical issue that needs to be fixed")

    total_chunks = 0
    ingestion_summary = {}

    for filename, text in docs.items():
        try:
            logger.info(f"[INGESTION] Processing: {filename}")
            logger.info(f"[INGESTION] Text length: {len(text)} characters")
            chunks = chunk_text(text)
            num_chunks = len(chunks)
            total_chunks += num_chunks
            logger.info(f"[INGESTION] Generated: {num_chunks} chunks")
            if num_chunks == 0:
                logger.warning(f"[INGESTION] ⚠️ No chunks generated for {filename}, skipping")
                ingestion_summary[filename] = {"chunks": 0, "status": "skipped"}
                continue
            metadatas = [{"source": filename}] * num_chunks
            logger.info(f"[INGESTION] Metadata structure: {{'source': '{filename}'}} (will be nested under 'metadata')")
            logger.info(f"[INGESTION] Generating embeddings and upserting to Qdrant...")
            vect.add_texts(
                texts=chunks,
                metadatas=metadatas
            )
            logger.info(f"[INGESTION] ✅ Successfully ingested {num_chunks} chunks from {filename}")
            ingestion_summary[filename] = {"chunks": num_chunks, "status": "success"}
        except Exception as e:
            logger.error(f"[INGESTION] ❌ Failed to ingest {filename}: {str(e)}", exc_info=True)
            ingestion_summary[filename] = {"chunks": 0, "status": "failed", "error": str(e)}

    try:
        logger.info(f"[INGESTION] Verifying ingestion...")
        collection_info = client.get_collection(collection_name=session_id)
        logger.info(f"[INGESTION] Collection stats:")
        logger.info(f"[INGESTION] Total points: {collection_info.points_count}")
        logger.info(f"[INGESTION] Vector dimension: {collection_info.config.params.vectors.size}")
        points, _ = client.scroll(
            collection_name=session_id,
            limit=3,
            with_payload=True,
            with_vectors=False
        )
        if points:
            logger.info(f"[INGESTION] Sample payload verification:")
            for i, point in enumerate(points, 1):
                payload = point.payload
                if 'metadata' in payload and isinstance(payload.get('metadata'), dict):
                    if 'source' in payload['metadata']:
                        logger.info(f"[INGESTION] Sample {i}: ✅ 'metadata.source' = '{payload['metadata']['source']}'")
                    else:
                        logger.warning(f"[INGESTION] Sample {i}: ⚠️ No 'source' in metadata")
                elif 'source' in payload:
                    logger.warning(f"[INGESTION] Sample {i}: ⚠️ 'source' at root level (unexpected for LangChain)")
                else:
                    logger.error(f"[INGESTION] Sample {i}: ❌ Unexpected payload structure: {list(payload.keys())}")
                content = payload.get('page_content', payload.get('text', ''))[:80]
                logger.debug(f"[INGESTION] Content preview: {content}...")
        else:
            logger.warning(f"[INGESTION] ⚠️ No points found in collection after ingestion!")
    except Exception as e:
        logger.error(f"[INGESTION] Error during verification: {e}")

    logger.info(f"{'='*80}")
    logger.info(f"[INGESTION] Ingestion Summary:")
    logger.info(f"[INGESTION] Session ID: {session_id}")
    logger.info(f"[INGESTION] Total documents: {len(docs)}")
    logger.info(f"[INGESTION] Total chunks: {total_chunks}")
    logger.info(f"[INGESTION] Per-file breakdown:")
    for filename, summary in ingestion_summary.items():
        status_emoji = "✅" if summary["status"] == "success" else "❌" if summary["status"] == "failed" else "⚠️"
        logger.info(f"[INGESTION] {status_emoji} {filename}: {summary['chunks']} chunks ({summary['status']})")
        if summary.get("error"):
            logger.info(f"[INGESTION] Error: {summary['error']}")
    logger.info(f"[INGESTION] NOTE: LangChain stores metadata as nested structure")
    logger.info(f"[INGESTION] Payload structure: {{'metadata': {{'source': filename}}, 'page_content': text}}")
    logger.info(f"[INGESTION] Filter key to use: 'metadata.source'")
    logger.info(f"{'='*80}")

    return ingestion_summary
