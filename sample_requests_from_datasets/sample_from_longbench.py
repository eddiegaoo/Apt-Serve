from datasets import load_dataset
import json
import random
from vllm.transformers_utils.tokenizer import get_tokenizer
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

parser=argparse.ArgumentParser()
parser.add_argument("--num-prompts", type=int, default=1000, help="Number of prompts to process.")
parser.add_argument("--model", type=str, default="facebook/opt-13b", required=True, help="Name of the model.")

datasets = ["gov_report", "qmsum", "multi_news", "vcsum"]

preprocessed_dataset: List[Tuple[str, int, int]] = []
tokenizer = get_tokenizer(args.model)

for dataset in datasets:
    data = load_dataset('THUDM/LongBench', dataset, split='test')
    for i in data:
        initial_prompt = 'Summarize the following: '
        prompt = initial_prompt + i['context']
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = i['answers'][0] #very akward, prompt is 'str' while 'answer' is list.
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
            #continue
        if prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue

        preprocessed_dataset.append((prompt, prompt_len, output_len))

print('number of queries left:{}'.format(len(preprocessed_dataset)))
for i in preprocessed_dataset:
    print('prompt_len:{}, output_len:{}'.format(i[1], i[2]))

if len(preprocessed_dataset) < args.num_prompts:
    times, extra = args.num_prompts // len(preprocessed_dataset), args.num_prompts % len(preprocessed_dataset)
    extra_data = random.sample(preprocessed_dataset, extra)
    preprocessed_dataset = times * preprocessed_dataset
    preprocessed_dataset.extend(extra_data)
    random.shuffle(preprocessed_dataset)
else:
    preprocessed_dataset = random.sample(preprocessed_dataset, args.num_prompts)
    random.shuffle(preprocessed_dataset)
    
data_filename = '../sampled_datasets/longbench.json'

with open(data_filename, 'w') as json_file:
    json.dump(preprocessed_dataset, json_file, indent=None)

    
    
