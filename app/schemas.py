from pydantic import BaseModel
from typing import List, Dict, Optional

# Existing RAG schemas (not changed)
class UploadResponse(BaseModel):
    filename: str
    status: str
    message: Optional[str] = None

class FileUploadResult(BaseModel):
    session_id: str
    results: List[UploadResponse]
    suggestions: List[str] = []

class QueryRequest(BaseModel):
    session_id: str
    query: str
    files: Optional[List[str]] = None

class QueryResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    source_docs: List[str] = []
    suggestions: List[str] = []

class MCQRequest(BaseModel):
    session_id: str
    num_questions: int = 5
    files: Optional[List[str]] = None

class SuggestionResponse(BaseModel):
    session_id: str
    suggestions: List[str]

class MCQItem(BaseModel):
    """Individual MCQ with question, choices, and correct answer"""
    question: str
    choices: Dict[str, str]  # {'A': "...", 'B': "...", 'C': "...", 'D': "..."}
    correct_answer: str

class MCQListResponse(BaseModel):
    session_id: str
    query: str
    mcqs: List[MCQItem]


# --- Session Management Schemas (with session_name) ---
class SessionCreateRequest(BaseModel):
    user_id: str
    session_name: Optional[str] = "Untitled Notebook"

class SessionCreateResponse(BaseModel):
    session_id: str
    user_id: str
    session_name: str

class SessionInfo(BaseModel):
    id: str
    user_id: str
    session_name: str
    created_at: str

class FileDeleteRequest(BaseModel):
    session_id: str
    filename: str
