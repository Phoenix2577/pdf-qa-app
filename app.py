import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
import chromadb
from chromadb.utils import embedding_functions
import openai

# Initialize OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in environment variables")
openai.api_key = OPENAI_API_KEY

# Initialize Chroma client
client = chromadb.Client()
collection = client.create_collection("pdf_collection")

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing; adjust in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper: Extract text from PDF
def extract_text_from_pdf(file: UploadFile) -> str:
    reader = PdfReader(file.file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# Helper: Ask OpenAI using context
def ask_openai(question: str, context: str) -> str:
    prompt = f"Use the following context to answer the question precisely.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=300,
        temperature=0
    )
    return response.choices[0].text.strip()

# Upload PDF and store in Chroma
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    text = extract_text_from_pdf(file)
    # Store as single document for simplicity
    collection.add(
        documents=[text],
        metadatas=[{"filename": file.filename}],
        ids=[file.filename]
    )
    return {"message": f"{file.filename} uploaded and indexed successfully."}

# Ask a question using all indexed PDFs
@app.post("/ask")
async def ask_question(question: str = Form(...)):
    # Retrieve all documents
    docs = collection.get(include=["documents"])["documents"]
    context = "\n".join([doc for doc in docs])
    answer = ask_openai(question, context)
    return {"answer": answer}

# Root endpoint
@app.get("/")
def root():
    return {"message": "Backend is live!"}

# Run server with dynamic Render PORT
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
