import json
import random
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import time
import re
import torch
import os
from dotenv import load_dotenv
import argparse

load_dotenv(dotenv_path="./checkmate/config.env")




parser = argparse.ArgumentParser()
parser.add_argument("--target_data", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
args = parser.parse_args()
veracity_prompt_path = "./checkmate/prompts/veracity_prompt.txt"

semantic_top_docs = args.target_data
output_file = args.output_file


VERACITY_MODEL = os.getenv("VERACITY_MODEL")
VERACITY_BATCH_SIZE = int(os.getenv("VERACITY_BATCH_SIZE"))
NUM_ANSWERS_PER_QUESTION = int(os.getenv("NUM_ANSWERS_PER_QUESTION"))  

print(VERACITY_BATCH_SIZE, "vbatch")
with open(semantic_top_docs) as f:
    claim_data = json.load(f)
    
total_claims = len(claim_data)
print(f"Total claims to process: {total_claims}")


llm = LLM(
        model=VERACITY_MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(VERACITY_MODEL)
sampling_params = SamplingParams(temperature=0.9,
        top_p=0.7, top_k=1,
         repetition_penalty=1.05, max_tokens=16000)


def prompt_llm(messages):
    start_time = time.time()
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    outputs = llm.generate([text], sampling_params)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.4f} seconds")
    return outputs





def create_final_prompt(claim_obj):
    # Load the base system prompt
    with open(veracity_prompt_path, "r", encoding="utf-8") as f:
        base_prompt = f.read()
    
    # Pull the claim and its associated best documents
    claim_text = claim_obj['claim']
    top_sentences = claim_obj['best_docs']

    # qa_pairs = []
    # for idx, (q, answers) in enumerate(top_sentences.items(), start=1):
    #     qa_block = f"Q{idx}: {q}\n"
    #     # Filter answers by score threshold
    #     valid_answers = [a for a in answers if a.get("score", 1.0) >= 0.52]
        
    #     if not valid_answers:
    #         qa_block += "No answer found.\n"
    #     else:
    #         for i in range(min(NUM_ANSWERS_PER_QUESTION, len(valid_answers))):
    #             text = valid_answers[i]["text"]
    #             url = valid_answers[i].get("url") or "URL not available"
    #             qa_block += f"Answer {i+1}: {text}\nSource: {url}\n\n"
        
    #     qa_pairs.append(qa_block.strip())
    qa_pairs = []
    for idx, (q, answers) in enumerate(top_sentences.items(), start=1):
        qa_block = f"Q{idx}: {q}\n"
        for i in range(min(NUM_ANSWERS_PER_QUESTION, len(answers))):
            text = answers[i]["text"]
            url = answers[i].get("url") or "URL not available"
            qa_block += f"Answer {i+1}: {text}\nSource: {url}\n\n"
        qa_pairs.append(qa_block.strip())


    qa_block = "\n\n".join(qa_pairs)
    base_prompt = base_prompt.replace("<claim>", claim_text).replace("<Question answer pairs>", qa_block)

    with open("final_prompt.txt", "w", encoding="utf-8") as f_out:
        f_out.write(base_prompt)
    # Construct the messages
    messages = [
        {
            "role": "system",
            "content": base_prompt
        },
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    

def parse_llm_response(llm_resp, batch_idx):
    output = llm_resp.outputs[0].text.strip()
    
    for line in output.splitlines():
        if "Label:" in line:
            # Extract everything after 'Label:'
            label = line.split("Label:", 1)[1].strip().lower()
            
            if "not enough evidence" in label:
                return "Not Enough Evidence"
            elif "conflicting" in label or "cherrypicking" in label:
                return "Conflicting Evidence/Cherrypicking"
            elif "supported" in label:
                return "Supported"
            elif "refuted" in label:
                return "Refuted"

    print("Going into fallback:", batch_idx)
    # fallback if structured label not found
    if "Not Enough Evidence" in output:
        return "Not Enough Evidence"
    elif any(x in output for x in ["Conflicting Evidence/Cherrypicking", "Cherrypicking", "Conflicting Evidence"]):
        return "Conflicting Evidence/Cherrypicking"
    elif any(x in output for x in ["Supported", "supported"]):
        return "Supported"
    elif any(x in output for x in ["Refuted", "refuted"]):
        return "Refuted"
    
    return None




def process_single_batch(batch, batch_idx):
    start_time = time.time()
    batch_prompts = [create_final_prompt(claim_obj) for claim_obj in batch]
    outputs = llm.generate(batch_prompts, sampling_params)        
    for i, claim_obj in enumerate(batch):
        claim_obj["final_pred"] = parse_llm_response(outputs[i], batch_idx)
        claim_obj["llm_output"] = outputs[i].outputs[0].text
    end_time = time.time()
    batch_time = end_time - start_time
    return batch, batch_time


batch_size = VERACITY_BATCH_SIZE

batches = [claim_data[i:i+batch_size] for i in range(0, len(claim_data), batch_size)]
print(f"\nProcessing {len(batches)} batches")

total_time = 0
start_time_all = time.time()
processed_claims = []
total_batches = len(batches)


for batch_idx, batch in enumerate(batches):
    batch_processed, batch_time = process_single_batch(batch, batch_idx)
    processed_claims.extend(batch_processed)
    total_time += batch_time
    print(f"Batch {batch_idx+1}/{total_batches} processed in {batch_time:.2f} seconds")

end_time_all = time.time()
total_time = end_time_all - start_time_all

time_per_claim = total_time / total_claims
time_per_batch = total_time / total_batches

print(f"\n--- Processing Summary ---")
print(f"Total time taken: {total_time:.2f} seconds")
print(f"Average time per batch: {time_per_batch:.2f} seconds")
print(f"Average time per claim: {time_per_claim:.4f} seconds")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(processed_claims, f, indent=4, ensure_ascii=False)

