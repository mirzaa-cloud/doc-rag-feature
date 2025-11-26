import json
from itertools import cycle
from fastapi import APIRouter, HTTPException
from app.schemas import QueryRequest, QueryResponse, SuggestionResponse, MCQRequest, MCQItem, MCQListResponse
from app.services.vecstore import get_qdrant_vectorstore, GroqEmbeddingsAdapter
from app.services.groq_embeddings import GroqEmbeddingService
from app.services.groq_llm import (
    GroqChatService,
    build_strict_context_prompt,
    build_suggestion_prompt,
    build_mcq_generation_prompt
)
from app.config import settings
from app.db.supabase_client import save_chat_message, get_session_messages
from typing import List, Optional, Mapping
from collections import defaultdict
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

router = APIRouter(prefix="/api/qa", tags=["qa"])
chat = GroqChatService()


def get_retriever_for_session(session_id: str, files: Optional[List[str]] = None, k: int = settings.TOP_K):
    """Get retriever with optional file filtering."""
    groq_service = GroqEmbeddingService()
    adapter = GroqEmbeddingsAdapter(groq_service)
    vect, client = get_qdrant_vectorstore(collection_name=session_id, adapter=adapter)

    if files:
        print(f"[FILTER] Creating retriever with file filter: {files}")

        # FIXED: Use metadata.source (LangChain nests metadata)
        conditions = [FieldCondition(key="metadata.source", match=MatchValue(value=f)) for f in files]

        # Use 'must' for single file, 'should' for multiple (OR logic)
        if len(files) == 1:
            metadata_filter = Filter(must=conditions)
            print(f"[FILTER] Using 'must' filter for single file: {files[0]}")
        else:
            metadata_filter = Filter(should=conditions)
            print(f"[FILTER] Using 'should' filter for {len(files)} files")

        retriever = vect.as_retriever(search_kwargs={"filter": metadata_filter, "k": k})
        print(f"[FILTER] Retriever created with metadata.source filter")
    else:
        print(f"[FILTER] Creating retriever without file filter (all documents)")
        retriever = vect.as_retriever(search_kwargs={"k": k})

    return retriever


def generate_suggestions_from_retriever(
    session_id: str,
    history: Optional[List[Mapping[str, str]]] = None,
    files: Optional[List[str]] = None
) -> List[str]:
    """Generate question suggestions based on documents and conversation history."""
    is_initial_suggestion = not history or len(history) < 2
    retrieval_k = 20 if is_initial_suggestion else settings.TOP_K

    retriever = get_retriever_for_session(session_id, files=files, k=retrieval_k)

    if is_initial_suggestion:
        search_query = "find the main topics, key concepts, and important entities from all uploaded files"
    else:
        all_user_queries = [msg.get('message') or msg.get('content', '') for msg in history if msg.get('role') == 'user']
        recent_queries = all_user_queries[-3:] if all_user_queries else []
        search_query = " ".join(recent_queries) if recent_queries else "find the main topics, key concepts, and important entities from all uploaded files"

    top_docs = retriever.invoke(search_query)

    if not top_docs:
        return []

    # Balance contexts across multiple files for initial suggestions
    if is_initial_suggestion:
        docs_by_source = defaultdict(list)
        for d in top_docs:
            source = d.metadata.get('source', 'Unknown File')
            docs_by_source[source].append(d.page_content)

        max_chunks_per_file = 10
        max_total_chunks = 30

        for source in docs_by_source:
            docs_by_source[source] = docs_by_source[source][:max_chunks_per_file]

        sources_cycle = cycle(docs_by_source.keys())
        balanced_contexts = []
        added = True

        while added and len(balanced_contexts) < max_total_chunks:
            added = False
            for _ in range(len(docs_by_source)):
                src = next(sources_cycle)
                if docs_by_source[src]:
                    balanced_contexts.append(docs_by_source[src].pop(0))
                    added = True
                    if len(balanced_contexts) >= max_total_chunks:
                        break

        contexts = balanced_contexts
    else:
        contexts = [d.page_content for d in top_docs]

    context_text = "\n-----DOCUMENT-----\n".join(contexts)

    suggestion_prompt = build_suggestion_prompt(
        context=context_text,
        history=history,
        all_previous_queries=None
    )

    resp = chat.chat_get_text(
        settings.GROQ_RAG_MODEL,
        suggestion_prompt,
        max_tokens=200,
        temperature=0.7
    )

    suggestions = []
    for line in resp.splitlines():
        line = line.strip()
        if not line:
            continue
        cleaned = line.lstrip('0123456789.-•) \t')
        if cleaned:
            suggestions.append(cleaned)

    return suggestions[:3]

@router.post("/query", response_model=QueryResponse)
def query_docs(body: QueryRequest):
    """Query documents with optional file filtering."""
    print(f"[QUERY] Session: {body.session_id}, Query: '{body.query}', Files: {body.files}")
    
    # Get retriever with file filter
    retriever = get_retriever_for_session(body.session_id, files=body.files, k=settings.TOP_K)
    top_docs = retriever.invoke(body.query)
    
    # Log what was retrieved
    print(f"[QUERY] Retrieved {len(top_docs)} documents")
    if top_docs:
        sources = {}
        for doc in top_docs:
            source = doc.metadata.get('source', 'UNKNOWN')
            sources[source] = sources.get(source, 0) + 1
        print(f"[QUERY] Source distribution: {sources}")
        
        if body.files:
            unexpected = [s for s in sources.keys() if s not in body.files]
            if unexpected:
                print(f"[QUERY] ⚠️  Got unexpected sources: {unexpected}")
            else:
                print(f"[QUERY] ✅ All results from specified files")
    
    # Save user query to history
    save_chat_message(body.session_id, "user", body.query)
    
    # Handle empty results
    if not top_docs:
        print(f"[QUERY] No documents found for query")
        try:
            suggestions = generate_suggestions_from_retriever(body.session_id, history=[], files=body.files)
        except Exception as e:
            print(f"[QUERY] Failed to generate suggestions: {e}")
            suggestions = []
        
        return QueryResponse(
            session_id=body.session_id,
            query=body.query,
            answer="I couldn't find relevant information in the specified documents. Please try a different query or check if the files are uploaded correctly.",
            source_docs=[],
            suggestions=suggestions
        )
    
    # Prepare contexts as LIST (not concatenated string!)
    contexts = [d.page_content for d in top_docs]
    filenames = [d.metadata.get('source', 'Unknown File') for d in top_docs]
    
    print(f"[QUERY] Prepared {len(contexts)} contexts from {len(set(filenames))} files")
    
    # Build prompt with correct parameter order (query first, contexts second)
    rag_prompt = build_strict_context_prompt(body.query, contexts)
    
    # Get answer from LLM
    print(f"[QUERY] Calling LLM with {settings.GROQ_RAG_MODEL}")
    answer = chat.chat_get_text(
        settings.GROQ_RAG_MODEL, 
        rag_prompt, 
        max_tokens=2048,
        temperature=0.1
    )
    
    print(f"[QUERY] Got answer: {len(answer)} chars")
    
    # Save assistant response to history
    save_chat_message(body.session_id, "assistant", answer, {"sources": list(set(filenames))})
    
    # Generate suggestions based on conversation history
    try:
        history = get_session_messages(body.session_id, limit=6)
        normalized_history = []
        for message in history:
            content = message.get('content') or message.get('text') or message.get('message') or ""
            normalized_history.append({
                'role': message.get('role'),
                'content': content
            })
        
        print(f"[QUERY] Generating suggestions with {len(normalized_history)} history messages")
        suggestions = generate_suggestions_from_retriever(
            body.session_id, 
            history=normalized_history, 
            files=body.files
        )
        print(f"[QUERY] Generated {len(suggestions)} suggestions")
    except Exception as e:
        print(f"[QUERY] ⚠️  Failed to generate suggestions: {e}")
        suggestions = []
    
    print(f"[QUERY] ✅ Returning answer with {len(list(set(filenames)))} sources and {len(suggestions)} suggestions")
    
    return QueryResponse(
        session_id=body.session_id,
        query=body.query,
        answer=answer,
        source_docs=list(set(filenames)),
        suggestions=suggestions
    )

@router.post("/suggestions", response_model=SuggestionResponse)
def generate_suggestions_api(body: QueryRequest):
    """Generate suggested questions."""
    suggestions = generate_suggestions_from_retriever(body.session_id, history=[], files=body.files)
    return SuggestionResponse(session_id=body.session_id, suggestions=suggestions)


@router.post("/generate-mcq", response_model=MCQListResponse)
def generate_mcq(body: MCQRequest):
    """Generate multiple choice questions from documents."""
    session_id = body.session_id
    count = body.num_questions or 5

    print(f"[MCQ] Session: {session_id}, Count: {count}, Files: {body.files}")

    # FIXED: Reduce k from 15 to 10 to avoid payload too large
    retriever = get_retriever_for_session(session_id, files=body.files, k=settings.TOP_K)
    docs = retriever.invoke("comprehensive overview topics concepts")

    # Log what was retrieved
    print(f"[MCQ] Retrieved {len(docs)} documents")
    if docs:
        sources = {}
        for doc in docs:
            source = doc.metadata.get('source', 'UNKNOWN')
            sources[source] = sources.get(source, 0) + 1
        print(f"[MCQ] Source distribution: {sources}")

        if body.files:
            unexpected = [s for s in sources.keys() if s not in body.files]
            if unexpected:
                print(f"[MCQ] ⚠️  Got unexpected sources: {unexpected}")
            else:
                print(f"[MCQ] ✅ All results from specified files")

    if not docs:
        print(f"[MCQ] ❌ No documents found!")
        raise HTTPException(status_code=404, detail="No documents found for MCQ generation")

    # FIXED: Take shorter excerpts to avoid payload too large
    by_source = defaultdict(list)
    for doc in docs:
        source = doc.metadata.get('source', 'unknown')
        # Take first 600 chars from each chunk (enough for MCQs)
        excerpt = doc.page_content[:600]
        by_source[source].append(excerpt)

    # Limit total number of excerpts
    selected_content = []
    sources_cycle = cycle(by_source.keys())
    max_excerpts = count * 2  # For 5 MCQs, use max 10 excerpts

    while len(selected_content) < max_excerpts and by_source:
        source = next(sources_cycle)
        if by_source[source]:
            selected_content.append(by_source[source].pop(0))
        else:
            del by_source[source]
            if not by_source:
                break

    combined_content = "\n\n".join(selected_content)

    # FIXED: Add final safety check to prevent payload too large
    MAX_CONTEXT = 5000
    if len(combined_content) > MAX_CONTEXT:
        print(f"[MCQ] ⚠️  Truncating context from {len(combined_content)} to {MAX_CONTEXT}")
        combined_content = combined_content[:MAX_CONTEXT]

    print(f"[MCQ] Final context: {len(combined_content)} chars for {count} MCQs")

    # Build MCQ prompt (contexts is now a list from build_mcq_generation_prompt)
    contexts = [combined_content]  # Wrap in list for compatibility
    mcq_prompt = build_mcq_generation_prompt(contexts, count)

    # Get MCQs from LLM
    print(f"[MCQ] Calling LLM for MCQ generation")
    resp = chat.chat_get_text(settings.GROQ_RAG_MODEL, mcq_prompt, max_tokens=4096, temperature=0.5)

    # FIXED: Better response cleaning
    print(f"[MCQ] Raw response length: {len(resp)}")
    print(f"[MCQ] Raw response preview: {resp[:200]}...")

    # Clean markdown code blocks if present
    resp = resp.strip()
    if resp.startswith('```'):
        lines = resp.split('\n')
        resp = '\n'.join(lines[1:])  # Remove first line (```json or ```)
    if resp.endswith('```'):
        resp = resp.rsplit('```', 1)[0]
    resp = resp.strip()

    # Try to find JSON array
    json_start = resp.find('[')
    json_end = resp.rfind(']') + 1
    if json_start != -1 and json_end > json_start:
        resp = resp[json_start:json_end]

    print(f"[MCQ] Cleaned response: {resp[:200]}...")

    # Parse MCQs
    try:
        mcqs_json = json.loads(resp)

        if not isinstance(mcqs_json, list):
            print(f"[MCQ] ❌ Response is not a list: {type(mcqs_json)}")
            raise ValueError("MCQ response is not a list")

        print(f"[MCQ] ✅ Successfully parsed {len(mcqs_json)} questions")

        # Validate and convert to MCQItem
        mcqs = []
        for idx, item in enumerate(mcqs_json[:count], 1):
            print(f"[MCQ] Question {idx}: {item.get('question', 'N/A')[:50]}...")

            mcq = MCQItem(
                question=item.get("question", ""),
                choices=item.get("choices", {}),
                correct_answer=item.get("correct_answer", "")
            )
            mcqs.append(mcq)

        print(f"[MCQ] Returning {len(mcqs)} validated MCQs")

        return MCQListResponse(
            session_id=session_id,
            query=f"Generate {count} MCQs",
            mcqs=mcqs
        )

    except json.JSONDecodeError as e:
        print(f"[MCQ] ❌ JSON decode error: {e}")
        print(f"[MCQ] Failed at position: {e.pos}")
        print(f"[MCQ] Context: {resp[max(0, e.pos-50):e.pos+50]}")
        raise HTTPException(status_code=500, detail=f"Failed to parse MCQ response: {str(e)}")

    except Exception as e:
        print(f"[MCQ] ❌ Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating MCQs: {str(e)}")
