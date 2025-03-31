#!/bin/bash

# Get the vllm package source directory using Python
vllm_dir=$(python3 -c "import importlib.util; spec = importlib.util.find_spec('vllm'); print(spec.origin.rsplit('/', 1)[0])")

# Check if the directory was successfully located
if [ -z "$vllm_dir" ]; then
  echo "vllm package not found. Exiting."
  exit 1
fi

echo "vllm package found at: $vllm_dir"


# Replace sequence and block part
cp "./aptserve_block.py" "$vllm_dir/block.py"
cp "./aptserve_sequence.py" "$vllm_dir/sequence.py"


#Replace attention part
cp "./attention/aptserve_layer.py" "$vllm_dir/attention/sequence.py"
cp "./attention/backends/aptserve_abstract.py" "$vllm_dir/attention/backends/abstract.py"
cp "./attention/backends/aptserve_flash_attn.py" "$vllm_dir/attention/backends/flash_attn.py"

#Replace engine part
cp "./engine/aptserve_llm_engine.py" "$vllm_dir/engine/llm_engine.py"

#Replace core part
cp "./core/aptserve_block_manager.py" "$vllm_dir/core/block_manager_v1.py"
cp "./core/aptserve_interfaces.py" "$vllm_dir/core/interfaces.py"
cp "./core/aptserve_scheduler.py" "$vllm_dir/core/scheduler.py"

#Replace worker part
cp "./worker/aptserve_cache_engine.py" "$vllm_dir/worker/cache_engine.py"
cp "./worker/aptserve_model_runner.py" "$vllm_dir/worker/model_runner.py"
cp "./worker/aptserve_worker.py" "$vllm_dir/worker/worker.py"

#Replace model_executor part
cp "./model_executor/layers/aptserve_linear.py" "$vllm_dir/model_executor/layers/linear.py"
cp "./model_executor/models/aptserve_opt.py" "$vllm_dir/model_executor/models/opt.py"