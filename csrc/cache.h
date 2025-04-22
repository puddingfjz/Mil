#pragma once

#include <torch/extension.h>

#include <map>
#include <vector>

void swap_blocks(
  torch::Tensor& src,
  torch::Tensor& dst,
  const std::map<int64_t, int64_t>& block_mapping);


void init_P2P_access(
  const int src_device_idx, 
  const int dst_device_idx,
  const int curr_device_idx);


void load_layer_weights(
  torch::Tensor& src,
  torch::Tensor& dst,
  const int layer_idx, 
  const int src_device_idx,
  const int dst_device_idx,
  const int curr_device_idx);

void disable_P2P_access(
  const int src_device_idx,
  const int dst_device_idx,
  const int curr_device_idx);



void copy_blocks(
  std::vector<torch::Tensor>& key_caches,
  std::vector<torch::Tensor>& value_caches,
  const std::map<int64_t, std::vector<int64_t>>& block_mapping);


// <>
void reorganize_blocks_deprecated(
  std::vector<torch::Tensor>& key_caches,
  std::vector<torch::Tensor>& value_caches,
  const std::map<int64_t, std::vector<int64_t>>& block_mapping, 
  std::vector<torch::Tensor>& new_key_caches,
  std::vector<torch::Tensor>& new_value_caches);


// <>
void reorganize_blocks(
  torch::Tensor& global_cache,
  std::vector<int64_t>& chains,
  std::vector<int64_t>& chain_lens,
  const int numel_per_block);


void reshape_and_cache(
  torch::Tensor& key,
  torch::Tensor& value,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  torch::Tensor& slot_mapping,
  const std::string& kv_cache_dtype);

void gather_cached_kv(
  torch::Tensor& key,
  torch::Tensor& value,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  torch::Tensor& slot_mapping);

// Just for unittest
void convert_fp8_e5m2(
  torch::Tensor& src_cache,
  torch::Tensor& dst_cache);
