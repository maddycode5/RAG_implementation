#                           //RETRIEVLAL CODE //  
# PDF LOADER DATA 
import os
import fitz

documents  = []

folder_path = r"C:\Users\Madhav\OneDrive\PROJECTS\Python_Project1\RAG_Implementation\pdfs"

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

document_texts = {}

for doc in documents:
    source = doc["source"]

    if source not in document_texts:          
        document_texts[source] = ""
            
    document_texts[source] += "\n" + doc["text"]

# create document name list
doc_names =list(
    document_texts.keys()
)

# pages = []

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
# document embeddings
doc_embeddings = model.encode(
    list(document_texts.values())
)
import numpy as np 
import faiss

# convert to numpy
doc_embeddings = np.array(doc_embeddings,
                          dtype = np.float32)

# build BM25 index
from rank_bm25 import BM25Okapi

tokenized_chunks = [
    item["text"].lower().split()
    for item in chunk_data 
]
#keyword search index
bm25 =BM25Okapi(tokenized_chunks)


# BRUTE FORCE METHOD
# storing the chunks and embeddings 
# vector_store = {
#     "chunks" : chunks,
#     "embeddings" : embeddings
# } 

embeddings = np.array(embeddings , dtype=np.float32)

faiss.normalize_L2(embeddings)
faiss.normalize_L2(doc_embeddings)

dimension = embeddings.shape[1]
# this FAISS returns cosine similarity scores 
index= faiss.IndexFlatIP(dimension)
index.add(embeddings)

# document FAISS Index
doc_dimension = doc_embeddings.shape[1]
doc_index = faiss.IndexFlatIP(doc_dimension)
doc_index.add(doc_embeddings)


from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    #Retrieval part of bm25
def bm25_retrieve(
        query,
        top_k = 20
):
    query_tokens=(
        query.lower().split()
    )
    scores = bm25.get_scores(
        query_tokens
    )

    top_indices = (
        np.argsort(scores)
        [-top_k:]
        [::-1]
    )

    return top_indices

def retrieve_documents(
        query,
        top_docs=3
):
    query_embedding = model.encode([query])

    query_embedding = np.array(
        query_embedding,
        dtype = np.float32
    )

    faiss.normalize_L2(query_embedding)


    scores,indices = doc_index.search(query_embedding ,
                                      top_docs)
    
    print("\nTop Documents: ")

    for score,idx in zip(scores[0],
                        indices[0]):
        print(
            f"{doc_names[idx]}  (score={score:.4f})"
            )
        
    return [
        doc_names[i]
        for i in indices[0]     
        ]
        
# retrieval part of Hybrid Search ( bm25 + faiss)
def retrieve(query,
            top_k = 20,
            sources = None):
    
    query_embedding = model.encode([query])

    query_embedding = np.array(
        query_embedding,
        dtype=np.float32
    )

    faiss.normalize_L2(query_embedding)
    
    # document filtering 
    candidate_chunk_indices = {
        i
        for i, chunk in enumerate(chunk_data)
        if sources is None or chunk["source"] in sources
    }
    # faiss retrieval
    faiss_scores, faiss_indices = index.search(
        query_embedding,
        100
    )

    filtered_faiss = []

    for idx in faiss_indices[0]:
        if idx in candidate_chunk_indices:
            filtered_faiss.append(idx)

    filtered_faiss = filtered_faiss[:top_k]

    # BM25 retrieval
    bm_indices = [
        idx
        for idx in bm25_retrieve(query, top_k * 3)
        if idx in candidate_chunk_indices
    ][:top_k]


    # hybrid merge 
    candidate_indices = list(
    set(
        filtered_faiss
        +
        bm_indices
    ))
    
    if len(candidate_indices) == 0:
        return None
    
    print("\nTop    Documents: ")
    print(sources)

    print("\nCombined Candidates: ")
    for idx in candidate_indices:
        print(
            chunk_data[idx]['source'],
            chunk_data[idx]["page"]
        )

    return [
        chunk_data[i]
        for i in candidate_indices
    ]

    #    // GENERATION CODE //

import requests
while True:
    question = input("\nAsk a question: ")

    if question.lower() == "exit":
        break
    
    # context = "\n".join(retrieved_chunks)
    # won't work because now retrieved chunks are dictionaries.

    top_docs = retrieve_documents(
        question,
        top_docs=3)
    
    retrieved_chunks = retrieve(question,
                                top_k =20,
                                sources = top_docs
                                )
    
    if retrieved_chunks is None:
        print("I couldn't find relevant information in the PDF.")
        continue

    pairs =[
        (question,
         chunk["text"]
        )
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

    for item in top_chunks:
        print(
            f"{item['source']} | Page {item['page']}"
        )
