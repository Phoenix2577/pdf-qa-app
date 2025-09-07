from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import PyPDF2
import faiss
import numpy as np
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store embeddings
index = None
chunks = []

def pdf_to_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_texts(texts):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=texts
    )
    return [np.array(d.embedding, dtype="float32") for d in response.data]

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile):
    global index, chunks
    text = pdf_to_text(file.file)
    chunks = chunk_text(text)

    embeddings = embed_texts(chunks)
    dim = len(embeddings[0])

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return {"message": "PDF processed successfully", "chunks": len(chunks)}

class Query(BaseModel):
    question: str

@app.post("/ask/")
async def ask(query: Query):
    global index, chunks
    if index is None:
        return {"answer": "Please upload a PDF first."}

    q_embed = embed_texts([query.question])[0].reshape(1, -1)
    D, I = index.search(q_embed, 3)  # Top 3 chunks
    retrieved = [chunks[i] for i in I[0]]

    context = "\n".join(retrieved)
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the context to answer."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query.question}"}
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=500,
        temperature=0
    )

    return {"answer": response.choices[0].message.content.strip()}
