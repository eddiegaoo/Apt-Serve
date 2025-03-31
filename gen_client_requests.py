"""Modified from vLLM's 'benchmark_serving.py' script.

On the server side, run one of the following commands:
    vLLM OpenAI API server
    python -m vllm.entrypoints.openai.api_server \
        --model <your_model> \
        --disable-log-requests


On the client side, run:
    python gen_client_requests.py \
        --backend <backend> \
        --model <your_model> \
        --dataset-name sharegpt \
        --dataset-path <path to dataset> \
        --request-rate <request_rate> \ # By default <request_rate> is inf
        --num-prompts <num_prompts> # 
   
"""
import argparse
import asyncio
import json
import os
import random
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import numpy as np
from backend_request_func_SLO import (ASYNC_REQUEST_FUNCS, RequestFuncInput,
                                  RequestFuncOutput)
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    p99_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    p99_itl_ms: float



async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
    cv: int,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    
    num_reqs = len(input_requests)
    input_requests = iter(input_requests)
    gamma_shape = (1/cv)**2
    scale = 1/(request_rate*gamma_shape)
    
    
    intervals = []
    print('create arrivals...')
    for i in range(num_reqs):
        interval = np.random.gamma(gamma_shape, scale)
        intervals.append(interval)
    intervals = np.array(intervals)
    count = 0
    
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
    
        # Sample the request interval from the exponential distribution.
        interval = intervals[count]
        count += 1
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)

def calculate_SLO(all_result, ttft_slo, tbt_slo, num_requests):
     
    all_output_lens = np.array(all_result['output_lens'])
    all_ttfts = np.array(all_result['ttfts'])
    all_itls = all_result['itls']

    all_zero_index = np.argwhere(all_output_lens==0).squeeze()
    all_non_zero_index = ~ (all_output_lens==0)
    num_complete_requests = all_result['completed']
    num_valid_requests = sum(all_non_zero_index)
    #sum requests only get one output token, or their services face a timeout.
    gap_ = num_complete_requests-num_valid_requests 

    all_output_lens = all_output_lens[all_non_zero_index]
    all_ttfts = all_ttfts[all_non_zero_index]

    all_itls_new = []
    for i in range(len(all_non_zero_index)):
        if all_non_zero_index[i]:
            all_itls_new.append(all_itls[i])
    all_itls = all_itls_new

    all_itls_ = []
    all_itls_new = []
    all_ttfts_new = []
    for i in range(len(all_itls)):
        if len(all_itls[i]) > 1:
            all_itls_.append(np.percentile(np.sort(all_itls[i][1:]),99))
            all_itls_new.append(all_itls[i][1:])
        else:
            all_itls_.append(0)
            all_itls_new.append(0)
        all_ttfts_new.append(all_ttfts[i])


    all_itls_ = np.array(all_itls_)
    all_itls = all_itls_new
    all_ttfts = np.array(all_ttfts_new)

    SLO_attainment = sum((all_ttfts<ttft_slo) & (all_itls_<tbt_slo)) + gap_

    TTFT_attainment = sum(all_ttfts<ttft_slo) + gap_
    TBT_attainment = sum(all_itls_<tbt_slo) + gap_

    print('\n')
    print("{s:{c}^{n}}".format(s=' SLO attainment Result ', n=50, c='='))
    print("{:<40} {:.2f}".format("Absolute SLO attainment:",SLO_attainment))
    print("{:<40} {:.2f}".format("Absolute TTFT attainment:",TTFT_attainment))
    print("{:<40} {:.2f}".format("Absolute P99 TBT attainment:",TBT_attainment))
    print("{:<40} {:.2f}".format("SLO attainment rate (%):",SLO_attainment/num_requests*100))
    print("{:<40} {:.2f}".format("TTFT attainment rate (%):",TTFT_attainment/num_requests*100))
    print("{:<40} {:.2f}".format("TBT attainment rate (%):",TBT_attainment/num_requests*100))
    print("=" * 50)
        
    

def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
) -> Tuple[BenchmarkMetrics, List[int]]:
    actual_output_lens: List[int] = []
    total_input = 0
    completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            # We use the tokenizer to count the number of output tokens for all
            # serving backends instead of looking at len(outputs[i].itl) since
            # multiple output tokens may be bundled together
            # Note : this may inflate the output token count slightly
            output_len = len(
                tokenizer(outputs[i].generated_text,
                          add_special_tokens=False).input_ids)
            actual_output_lens.append(output_len)
            total_input += input_requests[i][1]
            if output_len > 1:
                tpots.append(
                    (outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            completed += 1
        else:
            actual_output_lens.append(0)

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2)
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) *
        1000,  # ttfts is empty if streaming is not supported by backend
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        p99_tpot_ms=np.percentile(tpots or 0, 99) * 1000,
        mean_itl_ms=np.mean(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        p99_itl_ms=np.percentile(itls or 0, 99) * 1000,
    )

    return metrics, actual_output_lens


async def benchmark(
    backend: str,
    api_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int]],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
    cv: float,
    ttft_slo: float,
    tbt_slo: float,
    disable_tqdm: bool,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    print("Starting initial single prompt test run...")
    test_prompt, test_prompt_len, test_output_len = input_requests[0]
    num_requests = len(input_requests)
    test_input = RequestFuncInput(
        model=model_id,
        prompt=test_prompt,
        api_url=api_url,
        prompt_len=test_prompt_len,
        output_len=test_output_len,
        best_of=best_of,
        use_beam_search=use_beam_search,
    )
    test_output = await request_func(request_func_input=test_input)
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}")
    else:
        print("Initial test run completed. Starting main benchmark run...")
    print(f"Traffic request rate: {request_rate}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate, cv):
        prompt, prompt_len, output_len = request
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            best_of=best_of,
            use_beam_search=use_beam_search,
        )
        tasks.append(
            asyncio.create_task(
                request_func(request_func_input=request_func_input,
                             pbar=pbar)))
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
    )
    

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "std_ttft_ms": metrics.std_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "std_tpot_ms": metrics.std_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms,
        "mean_itl_ms": metrics.mean_itl_ms,
        "median_itl_ms": metrics.median_itl_ms,
        "std_itl_ms": metrics.std_itl_ms,
        "p99_itl_ms": metrics.p99_itl_ms,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
    }
    
    calculate_SLO(result, ttft_slo, tbt_slo, num_requests)
    
    return result


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    if args.base_url is not None:
        api_url = f"{args.base_url}:{args.port}{args.endpoint}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"

    tokenizer = get_tokenizer(tokenizer_id,
                              trust_remote_code=args.trust_remote_code)

    

    if args.dataset == "sharegpt":
        dataset_path = './sampled_datasets/sampled_sharegpt.json'
    if args.dataset == "humaneval":
        dataset_path = './sampled_datasets/sampled_humaneval.json'
    if args.dataset == "longbench":
        dataset_path = './sampled_datasets/sampled_longbench.json'
    
    with open(dataset_path) as f:
        input_requests = json.load(f)


    benchmark_result = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            model_id=model_id,
            tokenizer=tokenizer,
            input_requests=input_requests,
            best_of=args.best_of,
            use_beam_search=args.use_beam_search,
            request_rate=args.request_rate,
            cv=args.cv,
            ttft_slo=args.ttft_slo,
            tbt_slo=args.tbt_slo,
            disable_tqdm=args.disable_tqdm,
        ))

    # Save config and results to json
    if args.save_result:
        result_json: Dict[str, Any] = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["backend"] = backend
        result_json["model_id"] = model_id
        result_json["tokenizer_id"] = tokenizer_id
        result_json["best_of"] = args.best_of
        result_json["use_beam_search"] = args.use_beam_search
        result_json["num_prompts"] = args.num_prompts

        # Metadata
        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    kvstring = item.split("=")
                    result_json[kvstring[0].strip()] = kvstring[1].strip()
                else:
                    raise ValueError(
                        "Invalid metadata format. Please use KEY=VALUE format."
                    )

        # Traffic
        result_json["request_rate"] = (
            args.request_rate if args.request_rate < float("inf") else "inf")

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        # Save to file
        base_model_id = model_id.split("/")[-1]
        file_name = f"result-{args.request_rate}qps-{base_model_id}.json"  #noqa
        if args.result_filename:
            file_name = args.result_filename
        if args.result_dir:
            file_name = os.path.join(args.result_dir, file_name)
        with open(file_name, "w") as outfile:
            json.dump(result_json, outfile)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sharegpt",
        choices=["sharegpt","humaneval","longbench"],
        help="Name of the dataset to conduct serving simulation.",
    )
    
    parser.add_argument("--dataset-path",
                        type=str,
                        default='./dataset/path',
                        help="Path to the dataset.")
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-13b",
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help=
        "Name or path of the tokenizer, if not using the default tokenizer.",  # noqa: E501
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and "
        "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the ShareGPT dataset.")
    parser.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help=
        "Number of input tokens per request, used only for random sampling.",
    )
    parser.add_argument(
        "--ttft-slo",
        type=int,
        default=1.0,
        help=
        "Time to first token SLO.",
    )
    parser.add_argument(
        "--tbt-slo",
        type=int,
        default=1.0,
        help=
        "Time between tokens SLO.",
    )
    
    parser.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help=
        "Number of output tokens per request, used only for random sampling.",
    )
    parser.add_argument(
        "--random-range-ratio",
        type=float,
        default=1.0,
        help="Range of sampled ratio of input/output length, "
        "used only for random sampling.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=int("1"),
        help="The co-variance of gamma distribution (controls the burstiness)",
    )
    
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="Specify the filename to save benchmark json results."
        "If not specified, results will be saved in "
        "{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
        " format.",
    )

    args = parser.parse_args()
    main(args)
