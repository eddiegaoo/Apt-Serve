from datasets import load_dataset
import json
import random
from vllm.transformers_utils.tokenizer import get_tokenizer
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

parser=argparse.ArgumentParser()
parser.add_argument("--num-prompts", type=int, default=1000, help="Number of prompts to process.")
parser.add_argument("--model", type=str, default="facebook/opt-13b", required=True, help="Name of the model.")


dataset = load_dataset("openai_humaneval")
dataset = dataset['test']

preprocessed_dataset: List[Tuple[str, int, int]] = []
tokenizer = get_tokenizer(args.model)

for k in range(10):
    for i in dataset:
        prompt = i['prompt']
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = i['canonical_solution']
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue

        preprocessed_dataset.append((prompt, prompt_len, output_len))
    
print('shuffling dataset...')
random.shuffle(preprocessed_dataset)
preprocessed_dataset = random.sample(preprocessed_dataset, args.num_prompts)

data_filename = '../sampled_datasets/sampled_humaneval.json'

with open(data_filename, 'w') as json_file:
    json.dump(preprocessed_dataset, json_file, indent=None)
   