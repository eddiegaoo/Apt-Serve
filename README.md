# Apt-Serve
Code repository for the SIGMOD 25 paper: "Apt-Serve: Adaptive Request Scheduling on Hybrid Cache for Scalable LLM Inference Serving". <br>
Apt-Serve is a serving framework prototype implemented on top of vLLM (release version: 0.5.0 post1). <br>
All the adds-on by the framework are located in the folder `additional_designs`.

## Getting Started
1. Install the backbone system (vLLM 0.5.0. post1) first. Following guidelines from https://github.com/vllm-project/vllm.
2. Insert the additional designs:
```
bash additional_designs/insert_designs.sh
```
3. Install the customized cuda kernels to support hybrid cache:
```
python additional_designs/mixed_cache_kernels/mixed_cache_setup.py build_ext --inplace
```
With all these steps completed, the necessary implementation for the new designs has been integrated into vLLM and is ready for usage.

## Sample Serving Traces
Following `readme.md` from the folder `sample_requests_from_datasets` to sample requests to create a serving trace.
The sampled requests are automatically saved into `./sampled_datasets/` folder.

## Serving Simulation
Use OPT-13B as an example. <br>
Start the server side by:
```
python -m vllm.entrypoints.openai.api_server --model facebook/opt-13b --enforce-eager --disable-log-requests
```
After the server side is set up, start the client side code to simulate the request arrivals:
```
python gen_client_requests.py --model facebook/opt-13b --request-rate 3 --cv 1 --dataset sharegpt
```

## Exemplar Serving Result Comparsion
vLLM:<br>
<img width="540" alt="截屏2025-03-31 13 27 24" src="https://github.com/user-attachments/assets/a577010a-4c6f-42a7-9932-7cd146273aad" /> <br>
Apt-Serve:<br>
<img width="545" alt="截屏2025-03-31 13 27 34" src="https://github.com/user-attachments/assets/e378f6f2-73eb-4782-a863-b6febfc5cf6d" />


