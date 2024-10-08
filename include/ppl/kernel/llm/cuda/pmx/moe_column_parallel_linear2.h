// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#ifndef __PPL_KERNEL_LLM_CUDA_PMX_MOE_COLUMN_PARALLEL_LINEAR2_H__
#define __PPL_KERNEL_LLM_CUDA_PMX_MOE_COLUMN_PARALLEL_LINEAR2_H__

#include "ppl/kernel/llm/cuda/common/general_include.h"

#include "ppl/common/cuda/nccl_utils.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

struct moe_column_parallel_linear_config {
    int32_t num_experts;
    int32_t M;   // int64_t?
    int32_t N_d;  // int64_t?
    int32_t K;
    int32_t gemm_groups;
    // int32_t nccl_size;

    uint64_t buffer_size;
    uint64_t matrix_c_size;
    int32_t problem_sizes;
    int32_t ptr_x_size;  // should be differnent for a/b/c/c
    int32_t stride_As_size;
    int32_t stride_Bs_size;
    int32_t stride_Cs_size;
    int32_t stride_Ds_size;
    void* device_buffer;
    void* matrix_ds;
    void* gather_buffer;

    std::vector<int8_t> host_buffer;

    size_t workspace_size;
    int32_t arguments_workspace_id;
};

ppl::common::RetCode moe_column_parallel_linear_input_size(
    const ppl::common::TensorShape* input_shape,
    const ppl::common::TensorShape* weight_shape,
    const int64_t in_features,
    const int64_t out_features,
    const bool gather_output,
    const ppl::common::NcclParam* nccl_param,
    moe_column_parallel_linear_config& config);

ppl::common::RetCode moe_column_parallel_grouped_gemm(
    const cudaStream_t stream,  // pass to cutlass::gemm?
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* offset_shape,
    const void* expert_offset_host,
    const ppl::common::TensorShape* weight_shape,
    const void* weight,
    const ppl::common::TensorShape* bias_shape,
    const void* bias,
    const ppl::common::NcclParam* nccl_param,
    const bool gather_output,
    int device_id,
    moe_column_parallel_linear_config& config,
    const ppl::common::TensorShape* output_shape,
    void* output);

ppl::common::RetCode moe_column_parallel_linear2(
    const cudaStream_t stream,
    const ppl::common::NcclParam* nccl_param,
    const bool gather_output,
    void * workspace,
    moe_column_parallel_linear_config& config,
    const ppl::common::TensorShape* output_shape,
    void* output);

}}}}}

#endif
