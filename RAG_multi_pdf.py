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

                                    # document retrieval
document_texts = {}

for doc in documents:
    source = doc["source"]

    if source not in document_texts:          
        document_texts[source] = ""
            
    document_texts[source] += "\n" + doc["text"]

# # create document name list
doc_names =sorted(
    document_texts.keys()
)

#################################################
# DOCUMENT SUMMARIES
#################################################
def generate_document_summary(text):

    prompt = f"""
You are a document understanding assistant.

Summarize the following document in about 100-150 words.

Focus on:

- What type of document it is
- Main topics
- Important entities (people, projects, organizations, technologies)
- Purpose of the document

Do NOT copy large sections.

Document:

{text[:5000]}
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    result = response.json()

    if "response" not in result:
        print(result)
        return text[:1500]

    return result["response"].strip()

document_summaries = {}

for source, text in document_texts.items():

    print(f"\nGenerating summary for {source}...")

    summary = generate_document_summary(text)

    document_summaries[source] = summary


#FOR DEBUGGING PURPOSES ONLY 
print("\nDOCUMENT SUMMARIES\n")

for source, summary in document_summaries.items():

    print("-" * 50)
    print(source)
    print(summary)

# pages = []
                                    # OLD and basic method of chunking
# CREATING CHUNKS
# fixed chunking method
# def chunk_text(text,chunk_size =400,overlap = 100):

#     return [
#         text[i:i+chunk_size]
#         for i in range(0,len(text),chunk_size - overlap)
#     ]

# RECURSIVE CHUNKING 
from langchain_text_splitters import RecursiveCharacterTextSplitter

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size =2000, 
    chunk_overlap =200)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap =100)


parent_chunks =[]
parent_id = 0

# generate parent chunks

for doc in documents:
    parents = parent_splitter.split_text(doc["text"])

    for parent in parents:
        parent_chunks.append({
            "id" : parent_id,
            "text": parent,
            "page": doc["page"],
            "source" : doc["source"]
        })
        parent_id += 1

child_chunks = []

for parent in parent_chunks:
    children =child_splitter.split_text(parent["text"])

    for child in children:
        child_chunks.append({
            "text" : child,
            "parent_id" : parent["id"],
            "source" : parent["source"],
            "page":parent["page"]
            })
        
# generate emebddings 
from sentence_transformers import SentenceTransformer 
model = SentenceTransformer("all-MiniLM-L6-v2")

# only child chunks go to FAISS

embeddings = model.encode(
    [chunk["text"] for chunk in child_chunks]
)

# # document embeddings
doc_embeddings = model.encode(
    list(document_summaries.values())
)
import numpy as np 
import faiss

# convert to numpy
embeddings = np.array(embeddings , dtype=np.float32)
doc_embeddings = np.array(doc_embeddings,
                          dtype = np.float32)

# # build BM25 index
from rank_bm25 import BM25Okapi

tokenized_chunks = [
    item["text"].lower().split()
    for item in child_chunks 
]

#keyword search index
bm25 =BM25Okapi(tokenized_chunks)


# BRUTE FORCE METHOD
# storing the chunks and embeddings 
# vector_store = {
#     "chunks" : chunks,
#     "embeddings" : embeddings
# } 

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

# debugging code
print(f"Total PDFs indexed: {len(doc_names)}")
print(f"Total chunks indexed: {len(child_chunks)}")

# lets generate query 
def generate_queries(question):

    prompt = f"""
    You are a search query generator.

    The user's question belongs to a Retrieval-Augmented Generation system.

    Generate exactly FOUR alternative search queries.

    Rules:

    - Preserve the original meaning.
    - Do NOT answer the question.
    - Do NOT explain anything.
    - Do NOT invent new topics.
    - Do NOT output numbering.
    - Do NOT output headings.
    - One query per line.
    - Keep each query under 12 words.

    Question:

    {question}
    """

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model":"llama3",
            "prompt":prompt,
            "stream":False
        }
    )

    result = response.json()

    generated_text = result["response"]

    raw_queries = generated_text.split("\n")

    queries = []
    for q in raw_queries:

        q = q.strip()
        
        if not q:
            continue

        if q.lower().startswith("here are"):
            continue

        if q.lower().startswith("sure"):
            continue

        if q.lower().startswith("certainly"):
            continue

        for i in range(1, 10):
            if q.startswith(f"{i}."):
                q = q[len(f"{i}."):].strip()

        if q not in queries:
            queries.append(q)

    queries.insert(0, question)

    queries = list(dict.fromkeys(queries))

    print("\nGenerated Queries:")

    for q in queries:
        print("-", q)

    return queries

# HyDE generator 
def generate_hyde(query):

    prompt = f"""
    
    You are generating a hypothetical document for semantic retrieval.
    Write ONE short paragraph (50–80 words) that could plausibly appear inside a document containing the answer.

    Rules:
    - Do NOT answer with certainty.
    - Use generic wording whenever information is unknown.
    - Do NOT invent names, numbers, dates, organizations, or locations.
    - Do NOT include explanations or opinions.
    - Write only the hypothetical document.

    Question:
    {query}

    Hypothetical Answer:
    """

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model":"llama3",
            "prompt":prompt,
            "stream":False
        }
    )

    result = response.json()

    if "response" not in result:
        print(result)
        return ""
    
    generated_text = result["response"]

    return generated_text.strip()

from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def route_query(query):
    q=query.lower()

    # resume related questions
    if any(word in q for word in [
        "cgpa",
        "percentage",
        "certification",
        "skill",
        "certificate",
        "experience",
        "education",
        "madhav",
        "resume"
    ]):
        return ["resume (4).pdf"]

    # pdl report questions
    elif any(word in q for word in[
        "pdl",
        "project",
        "group member",
        "esp32"
    ]):
        return ["PDL_REPORT.pdf"]
    
    # Python Lab questions
    elif any(word in q for word in [
        "experiment",
        "python",
        "grade program"
    ]):
        return ["Python 3rd Semester Continued.pdf"]

    # Curriculum / syllabus questions
    elif any(word in q for word in [
        "ece",
        "semester",
        "subject",
        "curriculum",
        "syllabus"
    ]):
        return ["z.pdf"]
    
    # search all docs
    return None

def retrieve_documents(
        query,
        top_docs=3
        ):
    hyde_text = generate_hyde(query)

    query_embedding = model.encode([hyde_text])

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
        

from collections import defaultdict

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

# retrieval part of Hybrid Search ( bm25 + faiss)
def retrieve_single_query(query,
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
        for i, chunk in enumerate(child_chunks)
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
    # reciprocal rank fusion
    # rrf_scores = defaultdict(float)
    # k=60

    # for rank,idx in enumerate(filtered_faiss):
    #     rrf_scores[idx] += (
    #         1/(k+rank+1)
    #     ) 

    # for rank,idx in enumerate(bm_indices):
    #     rrf_scores[idx] += (
    #         1/(k+rank+1)
    #     )
    
    # candidate_indices = sorted(
    #     rrf_scores.keys(),
    #     key =lambda x : rrf_scores[x],
    #     reverse=True
    # )

    # candidate_indices =candidate_indices[:20]

    # if len(candidate_indices) == 0:
    #     return None

    # print("\nRRF Ranking: ")
    # for idx in candidate_indices[:10]:
    #     print(
    #         chunk_data[idx]['source'],
    #         chunk_data[idx]["page"],
    #         f"{rrf_scores[idx]:.5f}"
    #     )

    return {
        "faiss": filtered_faiss,
        "bm25": bm_indices
    }

    # multi query retrieval
def multi_query_retrieve(
        question,
        top_k = 10,
        sources = None
    ):
    queries= generate_queries(question)

    print("\nGenerated Queries : ")
    for q in queries:
        print("-",q)

    # all_chunks=[]
    # for q in queries:
    #     chunks = retrieve_single_query(q,
    #                                    top_k=top_k,
    #                                    sources = sources)
    #     if chunks :
    #         all_chunks.extend(chunks)
    # all_chunks = remove_duplicates(all_chunks)

    # return all_chunks

    rrf_scores = defaultdict(float)
    k = 60

    for q in queries:

        result = retrieve_single_query(
            q,
            top_k=top_k,
            sources=sources
        )

        for rank, idx in enumerate(result["faiss"]):
            rrf_scores[idx] += 1/(k+rank+1)

        for rank, idx in enumerate(result["bm25"]):
            rrf_scores[idx] += 1/(k+rank+1)

    candidate_indices = sorted(rrf_scores.keys(),
                               key =lambda x :rrf_scores[x],
                               reverse = True)
    
    candidate_indices = candidate_indices[:30]

    print("\nFinal Multi Query RRF Ranking:")

    for idx in candidate_indices[:10]:

        print(
            child_chunks[idx]["source"],
            child_chunks[idx]["page"],
            rrf_scores[idx]
        )

    retrieved_children = [
        child_chunks[idx]
        for idx in candidate_indices
]
    
    retrieved_children = remove_duplicates(retrieved_children)
    return retrieved_children

# remove duplicate chunks 
def remove_duplicates(chunks):
    seen =set()
    unique_chunks =[]
    for chunk in chunks:
        key =(
            chunk["source"],
            chunk["page"],
            chunk["text"]
        )
        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)
    
    return unique_chunks


# Contextual Compression of Chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

compression_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

def split_parent_chunks(parent_chunks):

    mini_chunks = []

    for chunk in parent_chunks:

        parts = compression_splitter.split_text(
            chunk["text"]
        )

        for part in parts:

            mini_chunks.append({

                "text": part,

                "source": chunk["source"],

                "page": chunk["page"]

            })

    return mini_chunks

def compress_context(question, mini_chunks, top_k=5):

    # Encode the question
    question_embedding = model.encode([question])

    question_embedding = np.array(
        question_embedding,
        dtype=np.float32
    )

    faiss.normalize_L2(question_embedding)

    # Encode every mini chunk
    mini_embeddings = model.encode(
        [chunk["text"] for chunk in mini_chunks]
    )

    mini_embeddings = np.array(
        mini_embeddings,
        dtype=np.float32
    )

    faiss.normalize_L2(mini_embeddings)

    # Cosine similarity
    scores = np.dot(
        mini_embeddings,
        question_embedding.T
    ).flatten()

    # Sort by similarity
    top_indices = np.argsort(scores)[::-1][:top_k]

    compressed_chunks = [
        mini_chunks[i]
        for i in top_indices
    ]

    return compressed_chunks

# llm based context compression
def llm_context_compression(question, compressed_chunks):
    context = "\n\n".join(
    chunk["text"]
    for chunk in compressed_chunks
)
    prompt = f"""
    You are a Context Compression Assistant for a Retrieval-Augmented Generation (RAG) system.

    Question:
    {question}

    Retrieved Context:
    {context}

    Your task is to compress the retrieved context before it is sent to the final answering model.

    Rules:
    - Keep ONLY information that is useful for answering the question.
    - Remove unrelated information.
    - Remove duplicate or repetitive sentences.
    - Preserve the original wording whenever possible.
    - Do NOT summarize unless multiple sentences express the same fact.
    - Do NOT invent headings or section titles.
    - Do NOT explain anything.
    - Do NOT answer the question.
    - Do NOT add any information that is not present in the context.
    - Return ONLY the filtered context.

    Compressed Context:
    """

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )
    result = response.json()

    if "response" not in result:
        print(result)
        return compressed_chunks

    page = compressed_chunks[0]["page"]
    source = compressed_chunks[0]["source"]

    compressed_text = result["response"].strip()
    mini_texts = compression_splitter.split_text(compressed_text)

    llm_chunks = []

    for chunk in mini_texts:

        llm_chunks.append({
            "text": chunk,
            "page": page,
            "source": source
        })

    return llm_chunks

    #    // GENERATION CODE //

import requests
while True:
    question = input("\nAsk a question: ")

    if question.lower() == "exit":
        break
    
    # context = "\n".join(retrieved_chunks)
    # won't work because now retrieved chunks are dictionaries.
    top_docs = route_query(question)

    if top_docs is None:
        top_docs = retrieve_documents(question,
                                      top_docs = 2)
 
    print("/Routed Documents : ")
    print(top_docs)
    
    retrieved_children = multi_query_retrieve(question,
                                top_k =10,
                                sources = top_docs
                                )
    
    if retrieved_children is None:
        print("I couldn't find relevant information in the PDF.")
        continue
    
    pairs =[
        (question,
         chunk["text"]
        )
        for chunk in retrieved_children
    ]

    rerank_scores= reranker.predict(pairs)

    ranked_results = sorted(zip(rerank_scores,retrieved_children),
                                reverse =True,
                                key=lambda x:x[0]
                                )
    top_children = [
        chunk
        for score, chunk in ranked_results[:5]
    ]

    parent_ids = {
        child["parent_id"]
        for child in top_children
    }

    parent_scores = {}

    for score, child in ranked_results[:10]:

        pid = child["parent_id"]

        if pid not in parent_scores:
            parent_scores[pid] = score
        else:
            parent_scores[pid] = max(
                parent_scores[pid],
                score
            )

    retrieved_parents = sorted(
        [
            parent
            for parent in parent_chunks
            if parent["id"] in parent_scores
        ],
        key=lambda x: parent_scores[x["id"]],
        reverse=True
    )[:3]
        
    
    retrieved_parents = retrieved_parents[:3]
    print("\nTop Reranked Chunks:")

    for score, chunk in ranked_results[:10]:
        print(
            score,
            chunk["source"],
            chunk["page"]
        )

    mini_chunks = split_parent_chunks(retrieved_parents)
    
    print("\nMini Chunks Created:")

    for i, chunk in enumerate(mini_chunks):

        print("\n------------------")

        print(i)

        print(chunk["source"])

        print(chunk["page"])

        print(chunk["text"][:200])
    
    compressed_chunks = compress_context(
        question,
        mini_chunks,
        top_k=5
    )

    compressed_chunks = llm_context_compression(
        question,
        compressed_chunks
        )

    print("\nCompressed Chunks:\n")

    context = "\n".join(
        chunk["text"]
        for chunk in compressed_chunks
    )

    # for degugging purpose only 
    print("\nLLM Compressed Chunks:\n")

    for chunk in compressed_chunks:
        print("-" * 40)
        print(chunk["source"])
        print(chunk["page"])
        print(chunk["text"])

    prompt = f"""
    
   You are a Retrieval-Augmented Generation assistant.

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

    print("\n Selected Parents: ")
    for parent in retrieved_parents:
        print(parent["source"],
              parent["page"]
              )
        
    print("\n Sources : ")

    for item in retrieved_parents:
        print(
            f"{item['source']} | Page {item['page']}"
        )
