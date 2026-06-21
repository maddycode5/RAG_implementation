#                           //RETRIEVLAL CODE //  
# PDF LOADER DATA 
import os
import fitz

documents  = []

folder_path = r"C:\Users\Madhav\NLP\pdfs"

for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(folder_path ,filename)

        pdf = fitz.open(pdf_path)
        for page_num,page in enumerate(pdf):
            documents.append({
                "text" : page.get_text(),
                "page" : page_num + 1,
                "source" : filename
            })

        pdf.close()

pages = []

# CREATING CHUNKS
# fixed chunking method
# def chunk_text(text,chunk_size =400,overlap = 100):

#     return [
#         text[i:i+chunk_size]
#         for i in range(0,len(text),chunk_size - overlap)
#     ]

# RECURSIVE CHUNKING 
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size =400, 
    chunk_overlap =100)

chunk_data =[]

for doc in documents:
    page_chunks = splitter.split_text(doc["text"])

    for chunk in page_chunks:
        chunk_data.append({
            "text": chunk,
            "page": doc["page"],
            "source" : doc["source"]
        })

# generate emebddings 
from sentence_transformers import SentenceTransformer 
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(
    [item["text"] for item in chunk_data]
)

# BRUTE FORCE METHOD
# storing the chunks and embeddings 
# vector_store = {
#     "chunks" : chunks,
#     "embeddings" : embeddings
# } 

import numpy as np 
import faiss

embeddings = np.array(embeddings , dtype=np.float32)

faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]
# this FAISS returns cosine similarity scores 
index= faiss.IndexFlatIP(dimension)

index.add(embeddings)


from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# retrieval part
def retrieve(query,
            top_k = 20,
            source =None):
    
    query_embedding = model.encode([query])

    query_embedding = np.array(
        query_embedding,
        dtype=np.float32
    )

    faiss.normalize_L2(query_embedding)
 
    scores ,indices = index.search(
        query_embedding,
        top_k
    )

    if scores[0][0]<0.15:
        return None
    
    print("\nTop Retrieved Chunks: ")
    for score ,idx in zip(
        scores[0],
        indices[0]
        ):
        print(f"\nScore : {score :.4f}")
        print(f"Page: {chunk_data[idx]['source']}")
        print(chunk_data[idx]["page"])

    return [
        chunk_data[i]
        for i in indices[0]
    ]

            #    // GENERATION CODE //


import requests
while True:
    question = input("\nAsk a question: ")

    if question.lower() == "exit":
        break
    
    # context = "\n".join(retrieved_chunks)
    # won't work because now retrieved chunks are dictionaries.

    retrieved_chunks= retrieve(
        question,
        top_k=20)

    pairs =[
        (question , chunk["text"])
        for chunk in retrieved_chunks 
    ]

    rerank_scores= reranker.predict(pairs)

    ranked_results = sorted(zip(rerank_scores,retrieved_chunks),
                                reverse =True,
                                key=lambda x:x[0]
                                )
    
    top_chunks = [
        chunk 
        for score,chunk in ranked_results[:3]
    ]

    context = "\n".join(
        chunk["text"]
        for chunk in top_chunks
    )
    
    if retrieved_chunks is None:
        print("I couldn't find relevant information in the PDF.")
        continue
    prompt = f"""
    
   You are a resume question-answering assistant.

    Answer ONLY using information explicitly present in the context.

    Do not make assumptions.

    Use the strongest evidence from the context.

    If multiple pieces of evidence exist,   
    prefer the most directly relevant one.

    If the answer is not present in the context, reply:

    "I could not find that information in the PDF."

    Answer:


    Context:
    {context}

    Question:
    {question}
    """

    print("\nRetrieved Context:")
    print(context)
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )
    result = response.json()
    
    if "response" in result :
        print("\nAnswer:")
        print(result["response"])

    else: 
        print(result)

    print("\n Sources : ")

    for item in retrieved_chunks:
        print(
            f" {item["source"]} | Page {item['page']}"
        )