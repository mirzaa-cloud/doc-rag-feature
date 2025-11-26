from fastapi import FastAPI
from app.api import files, qa
from app.api.sessions import router as sessions_router
app = FastAPI(title="Bitlabs Jobs - Doc QA API")

app.include_router(files.router)
app.include_router(qa.router)
app.include_router(sessions_router)
@app.get("/")
def root():
    return {"status": "ok", "service": "bitlabs-doc-qa"}

