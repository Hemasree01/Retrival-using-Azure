import os
import fitz  # PyMuPDF
import faiss
import numpy as np
from flask import Flask, request, render_template, session
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load .env variables
load_dotenv()
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
embedding_model = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
chat_model = os.getenv("AZURE_CHAT_DEPLOYMENT")

# Initialize OpenAI client
client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=endpoint,
    api_version="2024-12-01-preview"
)

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Required for session
vector_store = None
documents = []
uploaded_file_name = None

# Helper: Extract and chunk PDF
def extract_chunks(pdf_file, chunk_size=500):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = " ".join(page.get_text() for page in doc)
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# Helper: Embed text chunks
def embed_chunks(chunks):
    embeddings = []
    for chunk in chunks:
        response = client.embeddings.create(
            model=embedding_model,
            input=[chunk]
        )
        vector = response.data[0].embedding
        embeddings.append(vector)
    return np.array(embeddings).astype("float32")

# Helper: Query FAISS
def query_vector_store(question):
    global vector_store, documents
    question_embedding = client.embeddings.create(
        model=embedding_model,
        input=[question]
    ).data[0].embedding
    D, I = vector_store.search(np.array([question_embedding]).astype("float32"), k=3)
    retrieved_chunks = [documents[i] for i in I[0]]
    print("\n🔍 Top 3 Retrieved Chunks:")
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"\nChunk {i}:\n{'-'*40}\n{chunk}\n{'-'*40}")
    return "\n".join(retrieved_chunks)

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    global vector_store, documents, uploaded_file_name
    if request.method == "GET":
        # Clear chat history on page refresh (GET request)
        session.pop("chat_history", None)
    if "chat_history" not in session:
        session["chat_history"] = []

    answer = None

    if request.method == "POST":
        if "pdf_file" in request.files:
            pdf_file = request.files["pdf_file"]
            uploaded_file_name = pdf_file.filename
            chunks = extract_chunks(pdf_file)
            documents = chunks
            embeddings = embed_chunks(chunks)
            vector_store = faiss.IndexFlatL2(embeddings.shape[1])
            vector_store.add(embeddings)
            answer = f"PDF '{uploaded_file_name}' processed and indexed!"
            session["chat_history"].append({"question": "Uploaded PDF", "answer": answer})
        elif "question" in request.form:
            question = request.form["question"]
            if vector_store is None:
                answer = "Please upload a PDF first."
            else:
                context = query_vector_store(question)
                response = client.chat.completions.create(
                    model=chat_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an AI assistant that helps users learn from the information found in the source material. Answer the query using only the sources provided below. Use bullets if the answer has multiple points. Answer ONLY with the facts listed in the list of sources below. Cite your source when you answer the question. If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below."
                        },
                        {
                            "role": "user",
                            "content": f"Context:\n{context}\n\nQuestion: {question}"
                        }
                    ]
                )
                
                answer = response.choices[0].message.content
                print("\n📚 CONTEXT USED FOR ANSWERING:")
                print("="*60)
                print(context)
                print("="*60)

                print("\n🤖 GPT-4o FULL RESPONSE:")
                print("="*60)
                print(answer)
                print("="*60)
                session["chat_history"].append({"question": question, "answer": answer})

        session.modified = True

    return render_template(
        "index.html",
        answer=answer,
        uploaded_file_name=uploaded_file_name,
        chat_history=session.get("chat_history", [])
    )

if __name__ == "__main__":
    app.run(debug=True)
