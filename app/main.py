from fastapi import FastAPI
from pydantic import BaseModel
from app.chatbot import RAGChatbot
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
@app.get("/")
def home():
    return {
        "message": "Teyzix Core RAG System Running"
    }

@app.get("/chat")
def chat(query: str):

# CORS (IMPORTANT for UI)
    app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot = RAGChatbot()

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
def chat(request: QueryRequest):
    response = chatbot.chat(request.query)
    return {"response": response}

@app.get("/health")
def health():
    return {"status": "running"}