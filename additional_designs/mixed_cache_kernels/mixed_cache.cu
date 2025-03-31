#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

template <typename scalar_t>
__global__ void cache_to_hidden_kernel(
    scalar_t* __restrict__ hidden,
    const scalar_t* __restrict__ paged_cache,
    const int64_t* __restrict__ slot_mapping,
    const int64_t* __restrict__ dst_idx,
    const int hidden_stride,
    const int slot_stride,
    const int hidden_size) {
    const int64_t token_idx = dst_idx[blockIdx.x];
    const int64_t slot_idx = slot_mapping[token_idx];

    if (slot_idx < 0) {
        return;
    }
    
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    const int64_t slot_offset = i % hidden_size; 
    const int64_t src_idx = slot_idx * slot_stride + slot_offset;
    const int64_t tgt_idx = token_idx * hidden_stride + i;
    hidden[tgt_idx] = paged_cache[src_idx];
  }
}  
  
  
template <typename scalar_t>
__global__ void hidden_to_cache_kernel(
    const scalar_t* __restrict__ hidden,
    scalar_t* __restrict__ paged_cache,
    const int64_t* __restrict__ slot_mapping,
    const int64_t* __restrict__ src_idx,
    const int64_t* __restrict__ target_slot_idx,
    const int hidden_stride,
    const int slot_stride,
    const int hidden_size) {
    const int64_t token_idx = src_idx[blockIdx.x];
    const int64_t slot_idx = slot_mapping[target_slot_idx[blockIdx.x]];

    if (slot_idx < 0) {
        return;
    }
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        const int64_t slot_offset = i % hidden_size;
        const int64_t src_idx = token_idx * hidden_stride + i;
        const int64_t tgt_idx = slot_idx * slot_stride + slot_offset;
        paged_cache[tgt_idx] = hidden[src_idx];
    }
}


template <typename scalar_t>
__global__ void mixed_kv_to_cache_kernel(
    const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ k_cache,      // [num_blocks, block_size, num_heads, head_size]
    scalar_t* __restrict__ v_cache,      // [num_blocks, block_size, num_heads, head_size]
    scalar_t* __restrict__ shared_k_cache,  // [2*num_blocks, block_size, num_heads, head_size]
    scalar_t* __restrict__ shared_v_cache,  // [2*num_blocks, block_size, num_heads, head_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int64_t* __restrict__ slot_mapping_shared,  // [num_hidden_tokens]
    const int64_t* __restrict__ kv_cache_use_tot, // [num_kv_tokens]
    const int64_t* __restrict__ hidden_cache_use_tot, // [num_hidden_tokens]
    const int num_kv_tokens, 
    const int block_stride, 
    const int key_stride, 
    const int value_stride,
    const int num_heads, 
    const int head_size, 
    const int block_size) {
    
    int64_t token_idx=-1;
    int64_t slot_idx=-1;
    
    if (blockIdx.x < num_kv_tokens) { // persistent kv to private cache.
        token_idx = kv_cache_use_tot[blockIdx.x]; 
        slot_idx = slot_mapping[token_idx];
    }
    if (blockIdx.x >= num_kv_tokens && blockIdx.x < 2 * num_kv_tokens) { // persistent kv to shared cache.
        token_idx = kv_cache_use_tot[blockIdx.x % num_kv_tokens];
        slot_idx = slot_mapping[token_idx];
    }
    if (blockIdx.x >= 2*num_kv_tokens) { // temporary kv to shared cache.
        const int64_t offset_4_shared_mapping = blockIdx.x % (2*num_kv_tokens);
        token_idx = hidden_cache_use_tot[offset_4_shared_mapping];
        slot_idx = slot_mapping_shared[offset_4_shared_mapping];
    }
    if (slot_idx < 0) {
    return;
    }
    
    const int64_t block_idx = slot_idx / block_size;
    const int64_t block_offset = slot_idx % block_size;
    const int n = num_heads * head_size;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int64_t src_key_idx = token_idx * key_stride + i;
        const int64_t src_value_idx = token_idx * value_stride + i;
        const int head_idx = i / head_size;
        const int head_offset = i % head_size;
        const int64_t tgt_value_idx = block_idx * block_stride +
                                      block_offset * num_heads * head_size +
                                      head_idx * head_size + head_offset;
        if (blockIdx.x < num_kv_tokens) {
            k_cache[tgt_value_idx] = key[src_key_idx];
            v_cache[tgt_value_idx] = value[src_value_idx];
        }
        if (blockIdx.x >= num_kv_tokens) {
            shared_k_cache[tgt_value_idx] = key[src_key_idx];
            shared_v_cache[tgt_value_idx] = value[src_value_idx];
        }
  }
}


void hidden_to_cache(
    torch::Tensor hidden,
    torch::Tensor paged_cache,
    torch::Tensor slot_mapping,
    torch::Tensor src_idx,
    torch::Tensor target_slot_idx) {
    
    int num_target_tokens = src_idx.size(0);
    int hidden_size = hidden.size(1);
    int hidden_stride = hidden.stride(0);
    int slot_stride = paged_cache.stride(2);

    dim3 grid(num_target_tokens);
    dim3 block(std::min(hidden_size, 512));
    const at::cuda::OptionalCUDAGuard device_guard(hidden.device());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        hidden.scalar_type(), "hidden_to_cache", ([&] {
            hidden_to_cache_kernel<scalar_t><<<grid, block>>>(
                hidden.data_ptr<scalar_t>(),
                paged_cache.data_ptr<scalar_t>(),
                slot_mapping.data_ptr<int64_t>(),
                src_idx.data_ptr<int64_t>(),
                target_slot_idx.data_ptr<int64_t>(),
                hidden_stride,
                slot_stride,
                hidden_size);
        }));
}


void cache_to_hidden(
    torch::Tensor hidden,
    torch::Tensor paged_cache,
    torch::Tensor slot_mapping,
    torch::Tensor dst_idx) {
    
    int num_target_tokens = dst_idx.size(0);
    int hidden_size = hidden.size(1);
    int hidden_stride = hidden.stride(0);
    int slot_stride = paged_cache.stride(2);

    dim3 grid(num_target_tokens);
    dim3 block(std::min(hidden_size, 512));
    const at::cuda::OptionalCUDAGuard device_guard(hidden.device());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        hidden.scalar_type(), "hidden_to_cache", ([&] {
            cache_to_hidden_kernel<scalar_t><<<grid, block>>>(
                hidden.data_ptr<scalar_t>(),
                paged_cache.data_ptr<scalar_t>(),
                slot_mapping.data_ptr<int64_t>(),
                dst_idx.data_ptr<int64_t>(),
                hidden_stride,
                slot_stride,
                hidden_size);
        }));
}


void mixed_kv_to_cache(
    torch::Tensor& key,      // [num_tokens, num_heads, head_size]
    torch::Tensor& value,    // [num_tokens, num_heads, head_size]
    torch::Tensor& k_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor& v_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor& shared_k_cache, // [2*num_blocks, block_size, num_heads, head_size]
    torch::Tensor& shared_v_cache, // [2*num_blocks, block_size, num_heads, head_size]
    torch::Tensor& slot_mapping,  // [num_tokens]
    torch::Tensor& slot_mapping_shared,  // [num_hidden_tokens]
    torch::Tensor& kv_cache_use_tot, // [num_kv_tokens]
    torch::Tensor& hidden_cache_use_tot // [num_hidden_tokens]
    ) {
  
    int num_kv_tokens = kv_cache_use_tot.size(0);
    int num_hidden_tokens = slot_mapping_shared.size(0);
    int tot_tokens = 2*num_kv_tokens+num_hidden_tokens;
    int num_heads = key.size(1);
    int head_size = key.size(2);
    int block_size = k_cache.size(1);

    int key_stride = key.stride(0);
    int value_stride = value.stride(0);
    int block_stride = k_cache.stride(0);
    TORCH_CHECK(k_cache.stride(0) == v_cache.stride(0));

    dim3 grid(tot_tokens);
    dim3 block(std::min(num_heads * head_size, 512));
    const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      key.scalar_type(), "mixed_kv_to_cache", [&] {
        mixed_kv_to_cache_kernel<scalar_t>
            <<<grid, block, 0, stream>>>(
                key.data_ptr<scalar_t>(), 
                value.data_ptr<scalar_t>(),
                k_cache.data_ptr<scalar_t>(), 
                v_cache.data_ptr<scalar_t>(),
                shared_k_cache.data_ptr<scalar_t>(), 
                shared_v_cache.data_ptr<scalar_t>(), 
                slot_mapping.data_ptr<int64_t>(),
                slot_mapping_shared.data_ptr<int64_t>(),
                kv_cache_use_tot.data_ptr<int64_t>(),
                hidden_cache_use_tot.data_ptr<int64_t>(),
                num_kv_tokens,
                block_stride, 
                key_stride,
                value_stride,
                num_heads, 
                head_size, 
                block_size);
      });
}

// Binding the function to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hidden_to_cache", &hidden_to_cache, "hidden2cache kernel launcher");
    m.def("cache_to_hidden", &cache_to_hidden, "cache2hidden kernel launcher");
    m.def("mixed_kv_to_cache", &mixed_kv_to_cache, "mixedkv2cache kernel launcher");
}

