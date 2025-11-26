from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List
from app.db.supabase_client import (
    save_chat_message,
    get_session_by_id,
    add_session_file,         # add this
    remove_session_file       # add this
)

from app.utils.extractor import extract_text_by_extension
from app.services.guardrails import validate_file
from app.services.ingest import ingest_documents
from app.config import settings
from app.schemas import FileUploadResult, UploadResponse, FileDeleteRequest
from app.db.supabase_client import save_chat_message, get_session_by_id
from app.api.qa import generate_suggestions_from_retriever
from app.services.vecstore import delete_documents_by_source

router = APIRouter(prefix="/api/files", tags=["files"])

@router.post("/upload", response_model=FileUploadResult)
async def upload_files(
    session_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    # ...rest of your code
    # session_id must be from a session previously created by /sessions/create
    session = get_session_by_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

    results, to_ingest = [], {}
    initial_suggestions = []

    for f in files:
        content = await f.read()
        ok, msg = validate_file(f.filename, content)
        if not ok:
            results.append(UploadResponse(filename=f.filename, status="rejected", message=msg))
            continue

        try:
            text = extract_text_by_extension(f.filename, content)
        except Exception as e:
            err = f"Extraction failed: {type(e).__name__}: {str(e)}"
            results.append(UploadResponse(filename=f.filename, status="rejected", message=err))
            continue

        to_ingest[f.filename] = text
        results.append(UploadResponse(filename=f.filename, status="accepted", message="OK"))
        save_chat_message(session_id, "system", f"Uploaded {f.filename}", {"status": "accepted"})

        add_session_file(session_id, f.filename)

    if to_ingest:
        try:
            ingest_documents(session_id, to_ingest)
            initial_suggestions = generate_suggestions_from_retriever(session_id)
        except Exception as e:
            print(f"Failed to ingest documents for session {session_id}: {e}")

    return FileUploadResult(session_id=session_id, results=results, suggestions=initial_suggestions)

@router.post("/delete")
def delete_file(body: FileDeleteRequest):
    session = get_session_by_id(body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {body.session_id} not found.")
    delete_documents_by_source(body.session_id, body.filename)
    remove_session_file(body.session_id, body.filename)
    save_chat_message(body.session_id, "system", f"Deleted {body.filename}", {"status": "deleted"})
    return {"session_id": body.session_id, "filename": body.filename, "status": "deleted"}
