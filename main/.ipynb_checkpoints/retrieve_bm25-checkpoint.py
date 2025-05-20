import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
import pandas as pd
from tqdm import tqdm
import argparse

# Load environment variables
load_dotenv(dotenv_path="./checkmate/config.env")

# Configuration from .env
TOKEN_SIZE = int(os.getenv("CHUNK_TOKEN_SIZE", 256))
MAX_THREADS = int(os.getenv("MAX_BM25_THREADS", 8))
MAX_PROCESSES = os.cpu_count()



parser = argparse.ArgumentParser()
parser.add_argument("--knowledge_store_dir", type=str, required=True)
parser.add_argument("--target_data", type=str, required=True)
parser.add_argument("--json_output", type=str, required=True)
args = parser.parse_args()


# Paths
ks_path = args.knowledge_store_dir
hyde_claims_path = args.target_data
output_file = args.json_output
output_dir = os.path.dirname(output_file)
os.makedirs(output_dir, exist_ok=True)

# Load claim data
with open(hyde_claims_path, "r", encoding="utf-8") as f:
    claim_data = json.load(f)

def chunk_docs(docstore):
    chunks = []
    for idx, doc in enumerate(docstore):
        buffer = ""
        for i, sentence in enumerate(doc["url2text"]):
            if i == len(doc["url2text"]) - 1 or len(buffer) + len(sentence) >= TOKEN_SIZE:
                context_before = ""
                if chunks and chunks[-1].metadata["url"] == doc["url"]:
                    chunks[-1].metadata["context_after"] = buffer
                    context_before = chunks[-1].page_content

                chunks.append(
                    Document(
                        page_content=buffer,
                        metadata={
                            "url": doc["url"],
                            "context_before": context_before,
                            "context_after": "",
                            "evidence_metadata": {
                                "type": doc["type"],
                                "idx": idx,
                                "url": doc["url"],
                            },
                        },
                    )
                )
                buffer = ""
            buffer += sentence + " "
    return chunks

def retrieve_chunks_parallel(retriever, queries):
    def retrieve(q):
        return retriever.get_relevant_documents(query=q["query"])

    all_retrieved_docs = {}
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        results = list(executor.map(retrieve, queries))

    for retrieved_chunks in results:
        for doc in retrieved_chunks:
            all_retrieved_docs[doc.page_content] = doc  # Deduplicate by content
    return all_retrieved_docs

def process_claim(args):
    claim_idx, claim_obj, ks_path = args
    start_time = time.time()
    
    ks_file = os.path.join(ks_path, f"{claim_idx}.json")
    docstore = []
    with open(ks_file, 'r', encoding='utf-8') as f:
        for line in f:
            docstore.append(json.loads(line))

    chunks = chunk_docs(docstore)
    retriever = BM25Retriever.from_documents(chunks, k=250)

    queries = [
        {"type": "qa", "query": qa["question"] + " " + qa["answer"]}
        for qa in claim_obj["hyde_article"]
    ]

    all_retrieved_docs = retrieve_chunks_parallel(retriever, queries)

    best_docs_serialized = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in all_retrieved_docs.values()
    ]

    claim_obj["best_docs"] = best_docs_serialized
    duration = time.time() - start_time
    print(f"Claim {claim_idx} processed in {duration:.2f}s")
    return claim_obj

def main():
    start_time = time.time()

    claim_inputs = [(i, claim_data[i], ks_path) for i in range(len(claim_data))]
    
    print(f"Starting BM25 processing for {len(claim_inputs)} claims using {MAX_PROCESSES} processes...")
    
    with ProcessPoolExecutor(max_workers=MAX_PROCESSES) as executor:
        all_claims = list(tqdm(executor.map(process_claim, claim_inputs), total=len(claim_inputs), desc="BM25 Processing"))
        
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_claims, f, ensure_ascii=False, indent=2)

    avg_time = (time.time() - start_time) / len(all_claims)
    print(f"\nAverage time per claim: {avg_time:.2f} seconds")

if __name__ == "__main__":
    main()
