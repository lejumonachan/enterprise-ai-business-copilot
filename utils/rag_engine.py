import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LOCAL_AI_MODEL = "google/flan-t5-base"

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

tokenizer = AutoTokenizer.from_pretrained(LOCAL_AI_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_AI_MODEL)


def generate_local_answer(prompt, max_tokens=300):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        num_beams=4,
        do_sample=False
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks


def get_embedding(text):
    embedding = embedding_model.encode(text)
    return np.array(embedding, dtype="float32")


def build_faiss_index(chunks):
    if not chunks:
        raise ValueError("No text chunks available for indexing.")

    embeddings = [get_embedding(chunk) for chunk in chunks]
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, chunks


def retrieve_context(query, index, chunks, top_k=4):
    query_embedding = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)

    retrieved_chunks = []

    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            retrieved_chunks.append(chunks[idx])

    return "\n\n".join(retrieved_chunks)


def rag_answer(question, index, chunks):
    context = retrieve_context(question, index, chunks)

    if not context.strip():
        return "I could not find relevant information in the uploaded document."

    prompt = f"""
You are an enterprise document analyst.

Use the retrieved document context to answer the question.

Retrieved Context:
{context}

Question:
{question}

Answer format:
- Direct answer
- Key supporting points
- Business/technical meaning
"""

    try:
        answer = generate_local_answer(prompt)

        if len(answer.strip()) < 40:
            return f"""
### Retrieved Document Answer

### Question
{question}

### Most Relevant Document Sections
{context}

### Interpretation
The retrieved sections above are the most relevant parts of the uploaded document. Use these sections to understand the answer to the question.
"""

        return answer

    except Exception:
        return f"""
### Relevant Document Sections

{context}

### Question
{question}
"""