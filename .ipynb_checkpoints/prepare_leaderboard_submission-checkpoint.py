# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# # Created by zd302 at 12/01/2025

# import csv
# import json
# import argparse
# import os
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv(dotenv_path="config.env")
# NUM_ANSWERS_PER_QUESTION = int(os.getenv("NUM_ANSWERS_PER_QUESTION", 8))

# def convert(file_json, system_name):
#     with open(file_json) as f:
#         samples = json.load(f)

#     new_samples = []
#     for i, sample in enumerate(samples):
#         claim = sample['claim']
#         label = sample['final_pred']
#         prediction_evidence = ""
#         for question, answers in sample.get('best_docs', {}).items():
#             answer_texts = [answer['text'] for answer in answers[:NUM_ANSWERS_PER_QUESTION]]
#             concatenated_answers = " ".join(answer_texts)
#             prediction_evidence += question + "\t\t\n" + concatenated_answers + "\t\t\n\n"

#         new_samples.append([i, claim, prediction_evidence, label, 'pred'])

#     os.makedirs("output", exist_ok=True)
#     output_path = f"output/submission.csv"
#     with open(output_path, mode="w", newline="") as file:
#         writer = csv.writer(file)
#         writer.writerow(["id", "claim", "evi", "label", "split"])
#         writer.writerows(new_samples)

#     print(f"{file_json} has been converted to {output_path}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--filename", required=True, help="Path to the input JSON file")
#     parser.add_argument("--system_name", default="checkmate", help="System name (optional)")
#     args = parser.parse_args()

#     convert(args.filename, args.system_name)
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by zd302 at 12/01/2025

import csv
import json
import argparse
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path="config.env")
NUM_ANSWERS_PER_QUESTION = int(os.getenv("NUM_ANSWERS_PER_QUESTION", 8))

def convert(file_json, system_name):
    with open(file_json) as f:
        samples = json.load(f)

    new_samples = []
    for i, sample in enumerate(samples):
        claim = sample['claim']
        label = sample['final_pred']
        prediction_evidence = ""
        for question, answers in sample.get('best_docs', {}).items():
            answer_texts = [answer['text'] for answer in answers[:NUM_ANSWERS_PER_QUESTION]]
            concatenated_answers = " ".join(answer_texts)
            prediction_evidence += question + "\t\t\n" + concatenated_answers + "\t\t\n\n"

        new_samples.append([i, claim, prediction_evidence, label, 'pred'])

    os.makedirs("leaderboard_submission", exist_ok=True)
    output_path = f"leaderboard_submission/submission.csv"
    with open(output_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "claim", "evi", "label", "split"])
        writer.writerows(new_samples)

    print(f"{file_json} has been converted to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Process annotation files')
    parser.add_argument('--filename', type=str, default='data_store/baseline/dev_veracity_prediction.json',
                        help='Dataset filename (default: dev)')
    parser.add_argument('--system_name', type=str, default='baseline',
                        help='System name (default: baseline)')
    args = parser.parse_args()

    convert(args.filename, args.system_name)
    print("Done.")

if __name__ == "__main__":
    main()