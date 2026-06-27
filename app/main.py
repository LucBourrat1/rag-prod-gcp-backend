import os

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.auth import get_current_user, verify_google_token
from app.rag import ask_question, get_user_documents, process_pdf
from app.storage import upload_pdf_to_gcs

app = FastAPI(title="RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        os.getenv("FRONTEND_URL", ""),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    document_ids: list[str]


class TokenRequest(BaseModel):
    token: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth/verify")
async def verify_token(request: TokenRequest):
    user_info = await verify_google_token(request.token)
    return {"email": user_info["email"], "name": user_info.get("name")}


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...), current_user: dict = Depends(get_current_user)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Seuls les PDFs sont acceptés")
    content = await file.read()
    user_email = current_user["email"]
    gcs_path = await upload_pdf_to_gcs(content, file.filename, user_email)
    doc_id = await process_pdf(content, file.filename, user_email, gcs_path)
    return {"doc_id": doc_id, "filename": file.filename, "status": "indexed"}


@app.get("/documents")
async def list_documents(current_user: dict = Depends(get_current_user)):
    docs = await get_user_documents(current_user["email"])
    return {"documents": docs}


@app.post("/chat")
async def chat(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    answer = await ask_question(
        question=request.question,
        document_ids=request.document_ids,
        user_email=current_user["email"],
    )
    return answer
