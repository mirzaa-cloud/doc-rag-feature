from supabase import create_client
from app.config import settings

supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

def save_chat_message(session_id: str, role: str, message: str, extra: dict = None):
    data = {
        "session_id": session_id,
        "role": role,
        "message": message,
        "extra": extra or {}
    }
    resp = supabase.table("chat_messages").insert(data).execute()
    return resp

def get_session_messages(session_id: str, limit: int = 100):
    resp = supabase.table("chat_messages")\
        .select("*")\
        .eq("session_id", session_id)\
        .order("created_at", desc=True)\
        .limit(limit)\
        .execute()
    return resp.data

# --- Session Management Methods (with session_name) ---
def create_session(session_id: str, user_id: str, session_name: str = "Untitled Notebook"):
    data = {
        "id": session_id,
        "user_id": user_id,
        "session_name": session_name
    }
    resp = supabase.table("chat_sessions").insert(data).execute()
    return resp

def get_user_sessions(user_id: str):
    resp = supabase.table("chat_sessions")\
        .select("*")\
        .eq("user_id", user_id)\
        .order("created_at", desc=True)\
        .execute()
    return resp.data

def get_session_by_id(session_id: str):
    resp = supabase.table("chat_sessions")\
        .select("*")\
        .eq("id", session_id)\
        .single()\
        .execute()
    return resp.data
def add_session_file(session_id: str, filename: str):
    """
    Insert a new file record linked to a session.
    """
    data = {
        "session_id": session_id,
        "filename": filename
    }
    resp = supabase.table("session_files").insert(data).execute()
    return resp

def remove_session_file(session_id: str, filename: str):
    """
    Delete a file record linked to a session.
    """
    resp = (
        supabase.table("session_files")
        .delete()
        .match({"session_id": session_id, "filename": filename})
        .execute()
    )
    return resp

def get_files_for_session(session_id: str):
    """
    Retrieve all filenames linked to the session.
    """
    resp = (
        supabase.table("session_files")
        .select("filename")
        .eq("session_id", session_id)
        .execute()
    )
    files = [item["filename"] for item in resp.data] if resp.data else []
    return files
