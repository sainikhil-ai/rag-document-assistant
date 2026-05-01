
from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_pipeline import RAGPipeline

app = FastAPI(title="RAG Document Assistant")
rag = RAGPipeline()

class QuestionRequest(BaseModel):
    question: str

@app.on_event("startup")
def load_docs():
    rag.add_documents([
        "Generative AI systems can summarize documents and answer questions.",
        "Retrieval Augmented Generation improves response grounding.",
        "FastAPI is commonly used to deploy AI services."
    ])

@app.post("/ask")
def ask(req: QuestionRequest):
    return rag.answer(req.question)
