import time
from datetime import datetime, timedelta
import json
import nltk
import random
import gc
import torch
from rank_bm25 import BM25Okapi
import multiprocessing
import time
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import ast
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import argparse

prompts_path = "./checkmate/prompts/hyde_prompt.txt"
load_dotenv(dotenv_path="./checkmate/config.env")
HYDE_MODEL =  os.getenv("HYDE_MODEL")
HYDE_BATCH_SIZE = int(os.getenv("HYDE_BATCH_SIZE"))

parser = argparse.ArgumentParser(description="HyDE FC generation script")
parser.add_argument('--target_data', type=str, required=True, help='Path to input JSON with claims')
parser.add_argument('--json_output', type=str, required=True, help='Path to save processed output JSON')
args = parser.parse_args()

claims_path = args.target_data
output_path = args.json_output
batch_size = HYDE_BATCH_SIZE


llm = LLM(model=HYDE_MODEL,
          trust_remote_code=True,
          gpu_memory_utilization=0.95,
          enforce_eager=True,
          dtype="float16"
)
tokenizer = AutoTokenizer.from_pretrained(HYDE_MODEL)
sampling_params = SamplingParams(temperature=1.2, top_p=0.3, repetition_penalty=1.05, max_tokens=600)

def parse_qa_pairs(input_string):
    # Split by "QA:" to separate the label from the QA content
    if "QA:" in input_string:
        _, qa_content = input_string.split("QA:", 1)
    else:
        qa_content = input_string

    # Split the QA pairs by "||"
    qa_entries = [entry.strip() for entry in qa_content.split("||") if entry.strip()]

    # Split each QA entry into question and answer
    qa_pairs = []
    for entry in qa_entries:
        if '?' in entry:
            question_part, answer_part = entry.split('?', 1)
            question = question_part.strip() + '?'
            answer = answer_part.strip()
            qa_pairs.append({'question': question, 'answer': answer})

    return qa_pairs




with open(claims_path, "r", encoding="utf-8") as f:
    claim_data = json.load(f)

def create_prompt(claim):
    with open(prompts_path, "r", encoding="utf-8") as f:
        base_prompt = f.read()
    claim_text = claim["claim"]
    date = claim["claim_date"]
    speaker = claim["speaker"]
    location = claim["location_ISO_code"]
    reporting_source = claim["reporting_source"]
    
    base_prompt = base_prompt.replace("<claim>", claim_text)
    base_prompt = base_prompt.replace("<date>", date)
    base_prompt = base_prompt.replace("<location>", location or "")
    base_prompt = base_prompt.replace("<speaker>", speaker or "")
    base_prompt = base_prompt.replace("<source>", reporting_source or "")
    messages = [{"role": "system", "content": base_prompt}]

     # {"role": "user", "content":claim_json}
    tokenized_prompt =  tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenized_prompt


def process_single_batch_or_claim(batch, batch_idx):
    start_time = time.time()
    batch_prompts = [create_prompt(claim_obj) for claim_obj in batch]
    outputs = llm.generate(batch_prompts, sampling_params)

    processed_batch = []
    failed_claims = []

    for i, claim_obj in enumerate(batch):
        hyde_article = parse_qa_pairs(outputs[i].outputs[0].text)
        claim_obj["hyde_article"] = hyde_article
        claim_obj["original_article"] = outputs[i].outputs[0].text

        if len(hyde_article) <= 1:
            failed_claims.append(claim_obj)
        else:
            processed_batch.append(claim_obj)

    batch_time = time.time() - start_time
    return processed_batch, failed_claims, batch_time


total_claims = len(claim_data)
print(f"Total claims to process: {total_claims}")

batches = []
for i in range(0, len(claim_data), batch_size):
    batches.append(claim_data[i:i + batch_size])
    
print(f"\nProcessing {len(batches)} batches")

total_time = 0
start_time_all = time.time()
processed_claims = []
total_batches = len(batches)


retry_queue = []

for batch_idx, batch in enumerate(batches):
    # if batch_idx > 5:
    #     break
    processed, failed, batch_time = process_single_batch_or_claim(batch, batch_idx)
    processed_claims.extend(processed)
    retry_queue.extend(failed)
    total_time += batch_time
    print(f"Batch {batch_idx+1}/{total_batches} processed in {batch_time:.2f} seconds, {len(failed)} failed")


max_retries = 5
retry_queue = list(retry_queue)  # make sure it's a list

for retry_num in range(1, max_retries + 1):
    if not retry_queue:
        break

    print(f"\nRetry attempt {retry_num}: Retrying {len(retry_queue)} failed claims...")
    reprocessed, failed_again, retry_time = process_single_batch_or_claim(retry_queue, f"retry_{retry_num}")
    processed_claims.extend(reprocessed)
    retry_queue = failed_again
    print(f"Retry {retry_num} done in {retry_time:.2f} seconds, {len(failed_again)} still failed.")


end_time_all = time.time()
total_time = end_time_all - start_time_all

time_per_claim = total_time / total_claims
time_per_batch = total_time / total_batches

print(f"\n--- Processing Summary ---")
print(f"Total time taken: {total_time:.2f} seconds")
print(f"Average time per batch: {time_per_batch:.2f} seconds")
print(f"Average time per claim: {time_per_claim:.4f} seconds")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(processed_claims, f, indent=4, ensure_ascii=False)

print(f"Processed claims saved to {output_path}")
