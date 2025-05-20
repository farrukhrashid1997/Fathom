import json
from sentence_transformers import SentenceTransformer
import numpy as np
import time
import copy  
import os
from dotenv import load_dotenv
import argparse

load_dotenv(dotenv_path="./checkmate/config.env")
parser = argparse.ArgumentParser()
parser.add_argument("--target_data", type=str, required=True)
parser.add_argument("--json_output", type=str, required=True)
args = parser.parse_args()

bm25_top_docs = args.target_data
output_file = args.json_output


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
SENTENCE_EMBEDDING_BATCH_SIZE = int(os.getenv("SENTENCE_EMBEDDING_BATCH_SIZE"))


embed_model = SentenceTransformer(EMBEDDING_MODEL,trust_remote_code=True, model_kwargs={'attn_implementation': 'eager'}).to("cuda")




with open(bm25_top_docs, "r", encoding="utf-8") as claim_with_top_sentences:
    claim_data = json.load(claim_with_top_sentences)


all_claims = []
total_time = 0
copied_claim_data = copy.deepcopy(claim_data)
num_claims = len(copied_claim_data)

for claim_idx, claim_obj in enumerate(copied_claim_data):
    start_time = time.time()
    best_docs = claim_obj["best_docs"]
    qa_pairs = claim_obj["hyde_article"]
    top_sentences_text = [ doc["metadata"]["context_before"] +  doc["page_content"]  + doc["metadata"]["context_after"]  for doc in best_docs]

    sentence_embeddings = embed_model.encode(
        top_sentences_text,
        batch_size=SENTENCE_EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    best_docs_dict = {}
    all_queries = []
    all_queries.extend([{"type": "qa", "question": qa["question"], "query": qa["question"] + " " + qa["answer"] } for qa in qa_pairs])

    for query_obj in all_queries:
        query_text = query_obj["query"]
        query_embedding = embed_model.encode(
            [query_text],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
            prompt_name="query"
        )[0] 

        similarity = embed_model.similarity(query_embedding.reshape(1, -1), sentence_embeddings).squeeze()
        similarity = similarity.detach().cpu().numpy()

        top_k_index = np.argsort(similarity)[::-1][:10]
        top_docs_with_scores = []
        for i in top_k_index:
            doc_obj = best_docs[i]
            score = float(similarity[i])
            top_docs_with_scores.append({
                "text":   doc_obj["metadata"]["context_before"]  + doc_obj["page_content"] +  doc_obj["metadata"]["context_after"], 
                "score": score, 
                "url": doc_obj["metadata"]["url"]
            })

        key = query_obj["question"]
        if key not in best_docs_dict:
            best_docs_dict[key] = top_docs_with_scores


    claim_obj["best_docs"] = best_docs_dict
    all_claims.append(claim_obj)

    end_time = time.time()
    print("Per claim time taken:", end_time - start_time)
    total_time += (end_time - start_time)

# Final stats
average_time = total_time / num_claims if num_claims > 0 else 0
print(f"Average time per claim: {average_time:.2f} seconds")

# Save results
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_claims, f, ensure_ascii=False, indent=4)
