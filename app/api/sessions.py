from fastapi import APIRouter, HTTPException
from app.schemas import (
    SessionCreateRequest, 
    SessionCreateResponse,
    SessionInfo
)
from app.db.supabase_client import create_session, get_user_sessions, get_session_by_id
from app.services.vecstore import create_qdrant_collection
from uuid import uuid4

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

@router.post("/create", response_model=SessionCreateResponse)
def create_new_session(body: SessionCreateRequest):
    session_id = str(uuid4())
    create_qdrant_collection(session_id)
    create_session(session_id, body.user_id, body.session_name)
    return SessionCreateResponse(
        session_id=session_id,
        user_id=body.user_id,
        session_name=body.session_name
    )

@router.get("/list/{user_id}")
def list_sessions(user_id: str):
    sessions = get_user_sessions(user_id)
    # If you use the SessionInfo schema elsewhere, convert dict -> SessionInfo if necessary
    return {"user_id": user_id, "sessions": sessions}

@router.get("/{session_id}")
def get_session_details(session_id: str):
    session = get_session_by_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session
