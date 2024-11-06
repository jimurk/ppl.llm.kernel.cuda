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

#include "ppl/kernel/llm/cuda/pmx/moe_column_parallel_linear2.h"

#include <vector>
#include <mutex>
#include <iostream>  // debug

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/util/packed_stride.hpp"

#include "cudakernel/memory/transpose.h"
#include "ppl/common/log.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

#define CEILING(number, grain, shift) (((number + grain - 1) >> (shift)) << (shift))
#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

using namespace cute;
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>; // <M,N,K> per group
using ElementA = cutlass::half_t;  // Element type for A matrix operand
using ElementB = cutlass::half_t;  // Element type for B matrix operand
using ElementC = cutlass::half_t;  // Element type for C and D matrix operands

// A matrix configuration
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration  // ColumnMajor?
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
// using         LayoutC     = cutlass::layout::ColumnMajor;  // ???        // Layout type for C and D matrix operands
using         LayoutC     = cutlass::layout::RowMajor;  // ???              // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Alignment of C matrix in units of elements (up to 16 bytes)

// Core kernel configurations
using ElementAccumulator  = float;      // ???                              // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm90;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
using TileShape           = Shape<_256,_128,_128>;                          // Threadblock-level tile size
using ClusterShape        = Shape<_2,_2,_1>;                                // Shape of the threadblocks in a cluster
using StageCountType = cutlass::gemm::collective::StageCountAuto;           // Stage count maximized based on the tile size
// using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum; // Kernel to launch
using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative; // Kernel to launch
using EpilogueSchedule = cutlass::epilogue::PtrArrayNoSmemWarpSpecialized;   // Epilogue to launch

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,  // OpClassSimt, cuda core ???
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC *, AlignmentC,
    ElementC, LayoutC *, AlignmentC,
    EpilogueSchedule
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA *, AlignmentA,
    ElementB, LayoutB *, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;

struct GemmArguments {
    Gemm::Arguments arguments;
    int32_t arguments_workspace_id;
};

static std::vector<GemmArguments> gemm_arguments;
static std::mutex gemm_locker;

int32_t enqueue_arguments(Gemm::Arguments &arguments) {
    std::lock_guard<std::mutex> lock_guard(gemm_locker);
    int32_t size = gemm_arguments.size();
    gemm_arguments.push_back({arguments, size});

    return size;
}

void remove_arguments(int32_t arguments_workspace_id) {
    std::lock_guard<std::mutex> lock_guard(gemm_locker);
    auto iterator = gemm_arguments.begin() + arguments_workspace_id;
    gemm_arguments.erase(iterator);
}

template <typename T>
__global__ void set_2D_by_1D(const T* input_1d,
                             int32_t* expert_offset, int32_t offset_size,
                             int M, int N, T* output_2d) {
    int element_x = blockIdx.x * blockDim.x + threadIdx.x;
    int element_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (element_x >= N || element_y >= M) {
        return;
    }

    int expert_id = -1;
    int start = 0;
    int end = 0;
    for (int i = 0; i < offset_size - 1; i++) {
        end = expert_offset[i + 1];
        if (element_y >= start && element_y < end) {
            expert_id = i;
            break;
        }
        start = end;
    }
    if (expert_id < 0 || expert_id >= offset_size - 1) {
        return;
    }

    T value = input_1d[expert_id * N + element_x];
    output_2d[element_y * N + element_x] = value;
}

template <typename T>
void set_matrix_c(const cudaStream_t stream, const T* bias, const int64_t* expert_offset_host, int32_t* buffer,
                  int32_t num_experts, int32_t M, int32_t N_d, T* output) {
    std::vector<int32_t> expert_offset;
    for (int32_t i = 0; i < num_experts + 1; i++) {
        expert_offset.push_back((int32_t)expert_offset_host[i]);
    }
    cudaMemcpy(buffer, expert_offset.data(), sizeof(int32_t) * (num_experts + 1), cudaMemcpyHostToDevice);

    dim3 grid, block;
    block.x = 64;
    block.y = 4;
    grid.x = (N_d + block.x - 1) / block.x;
    grid.y = (M + block.y - 1) / block.y;

    set_2D_by_1D<T><<<grid, block, 0, stream>>>(bias, buffer, num_experts + 1, M, N_d, output);
}

ppl::common::RetCode moe_column_parallel_linear_input_size(
    const ppl::common::TensorShape* input_shape,
    const ppl::common::TensorShape* weight_shape,
    const int64_t in_features,
    const int64_t out_features,
    const void* bias,
    const bool gather_output,
    const ppl::common::NcclParam* nccl_param,
    moe_column_parallel_linear_config& config)
{
    // std::cout << "come in input_size" << std::endl;
    if (input_shape == nullptr) {
        LOG(ERROR) << "input shape can not be null.";
        return ppl::common::RC_UNSUPPORTED;
    }
    if (weight_shape == nullptr) {
        LOG(ERROR) << "weight shape can not be null.";
        return ppl::common::RC_UNSUPPORTED;
    }
    int nccl_size;
    if (nccl_param == nullptr) {
        nccl_size = 1;
    }
    else {
        nccl_size = nccl_param->size;
    }

    config.num_experts = weight_shape->GetDim(0);
    config.M = input_shape->CalcElementsToDimensionExcludingPadding(input_shape->GetDimCount() - 1);
    config.N_d = out_features / nccl_size;
    config.K = in_features;
    config.gemm_groups = config.num_experts;

    uint64_t buffer_size = 0;
    uint64_t matrix_c_size = CEILING(config.M * config.N_d * sizeof(ElementC), 127, 7);
    if (bias != nullptr) {
        buffer_size += matrix_c_size;  // matrix c
    }
    config.matrix_c_size = matrix_c_size;

    if (gather_output && nccl_size > 1) {
        buffer_size += matrix_c_size * nccl_size;  // gather_buffer, namely, matrix_ds
    }

    int32_t problem_sizes = CEILING((sizeof(int32_t) * 3 * config.gemm_groups), 127, 7);
    buffer_size += problem_sizes;
    config.problem_sizes = problem_sizes;

    // ElementA*, ElementB*, ElementC* and ElementD* have the same size, 8 bytes.
    int32_t ptr_x_size = CEILING((sizeof(ElementA*) * config.gemm_groups), 127, 7);
    if (bias != nullptr) {
        buffer_size += ptr_x_size * 4;
    }
    else {
        buffer_size += ptr_x_size * 3;
    }
    config.ptr_x_size = ptr_x_size;

    int32_t stride_As_size = CEILING((sizeof(StrideA) * config.gemm_groups), 127, 7);
    buffer_size += stride_As_size;
    config.stride_As_size = stride_As_size;

    int32_t stride_Bs_size = CEILING((sizeof(StrideB) * config.gemm_groups), 127, 7);
    buffer_size += stride_Bs_size;
    config.stride_Bs_size = stride_Bs_size;

    int32_t stride_Cs_size = 0;
    if (bias != nullptr) {
        stride_Cs_size = CEILING((sizeof(StrideC) * config.gemm_groups), 127, 7);
    }
    buffer_size += stride_Cs_size;
    config.stride_Cs_size = stride_Cs_size;

    int32_t stride_Ds_size = CEILING((sizeof(StrideD) * config.gemm_groups), 127, 7);
    buffer_size += stride_Ds_size;
    config.stride_Ds_size = stride_Ds_size;
    config.buffer_size = buffer_size;

    std::cout << "config.num_experts: " << config.num_experts << std::endl;
    std::cout << "config.M: " << config.M << std::endl;
    std::cout << "config.N_d: " << config.N_d << std::endl;
    std::cout << "config.K: " << config.K << std::endl;
    std::cout << "config.matrix_c_size: " << config.matrix_c_size << std::endl;
    std::cout << "config.problem_sizes: " << config.problem_sizes << std::endl;
    std::cout << "config.ptr_x_size: " << config.ptr_x_size << std::endl;
    std::cout << "config.stride_As_size: " << config.stride_As_size << std::endl;
    std::cout << "config.stride_Bs_size: " << config.stride_Bs_size << std::endl;
    std::cout << "config.stride_Cs_size: " << config.stride_Cs_size << std::endl;
    std::cout << "config.stride_Ds_size: " << config.stride_Ds_size << std::endl;
    std::cout << "config.buffer_size: " << config.buffer_size << std::endl;

    return ppl::common::RC_SUCCESS;
}

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
    int device_id,  // use 0 as default, remove this parameter
    moe_column_parallel_linear_config& config,
    const ppl::common::TensorShape* output_shape,
    void* output)
{
    if (input_shape->GetDataType() != ppl::common::DATATYPE_FLOAT16 &&
        input_shape->GetDataType() != ppl::common::DATATYPE_FLOAT32 &&
        input_shape->GetDataType() != ppl::common::DATATYPE_INT8) {
        LOG(ERROR) << "supported data type of A matrix: float16, float32 and int8.";
        return ppl::common::RC_UNSUPPORTED;
    }
    if (weight_shape->GetDataType() != ppl::common::DATATYPE_FLOAT16 &&
        weight_shape->GetDataType() != ppl::common::DATATYPE_FLOAT32 &&
        weight_shape->GetDataType() != ppl::common::DATATYPE_INT8) {
        LOG(ERROR) << "supported data type of B matrix: float16, float32 and int8.";
        return ppl::common::RC_UNSUPPORTED;
    }
    if (bias_shape->GetDataType() != ppl::common::DATATYPE_FLOAT16 &&
        bias_shape->GetDataType() != ppl::common::DATATYPE_FLOAT32 &&
        bias_shape->GetDataType() != ppl::common::DATATYPE_INT32) {
        LOG(ERROR) << "supported data type of C matrix: float16, float32 and int32.";
        LOG(ERROR) << "only support fp16 C matrix.";
        return ppl::common::RC_UNSUPPORTED;
    }
    if (output_shape->GetDataType() != ppl::common::DATATYPE_FLOAT16 &&
        output_shape->GetDataType() != ppl::common::DATATYPE_FLOAT32 &&
        output_shape->GetDataType() != ppl::common::DATATYPE_INT8 &&
        output_shape->GetDataType() != ppl::common::DATATYPE_INT32) {
        LOG(ERROR) << "supported data type of D matrix: float16, float32, int8 and int32.";
        return ppl::common::RC_UNSUPPORTED;
    }
    if (offset_shape->GetDataType() != ppl::common::DATATYPE_INT64) {
        LOG(ERROR) << "only support int64 group offset.";
        return ppl::common::RC_UNSUPPORTED;
    }

    ElementC* matrix_cs = nullptr;
    if (bias != nullptr) {
        matrix_cs = (ElementC*)config.device_buffer;
    }
    void* gather_buffer = nullptr;
    ElementC* matrix_ds = nullptr;
    int32_t* problem_sizes_device = nullptr;
    if (gather_output && nccl_param->size > 1) {
        if (bias != nullptr) {
            gather_buffer = (void*)((uint8_t*)config.device_buffer + config.matrix_c_size);
        }
        matrix_ds = (ElementC*)((uint8_t*)gather_buffer + config.matrix_c_size * nccl_param->rank);
        problem_sizes_device = (int32_t*)((uint8_t*)gather_buffer + config.matrix_c_size * nccl_param->size);
    }
    else {
        matrix_ds = (ElementC*)output;
        if (bias != nullptr) {
            problem_sizes_device = (int32_t*)((uint8_t*)matrix_cs + config.matrix_c_size);
        }
        else {
            problem_sizes_device = (int32_t*)config.device_buffer;
        }
    }
    config.gather_buffer = gather_buffer;
    config.matrix_ds = (void*)matrix_ds;
    // if (bias == nullptr) {
        // cudaMemset((void *)matrix_cs, config.matrix_c_size, 0);
    // }
    if (bias != nullptr) {
        set_matrix_c(stream, (ElementC*)bias, (const int64_t*)expert_offset_host,
                     (int32_t*)matrix_ds, config.num_experts, config.M,
                     config.N_d, (ElementC*)matrix_cs);
    }
    const ElementA** ptr_As_device = (const ElementA**)((uint8_t*)problem_sizes_device + config.problem_sizes);
    const ElementB** ptr_Bs_device = (const ElementB**)((uint8_t*)ptr_As_device + config.ptr_x_size);
    const ElementC** ptr_Cs_device = (const ElementC**)((uint8_t*)ptr_Bs_device + config.ptr_x_size);
    ElementC** ptr_Ds_device = (ElementC**)((uint8_t*)ptr_Cs_device + config.ptr_x_size);
    StrideA* stride_A_device = (StrideA*)((uint8_t*)ptr_Ds_device + config.ptr_x_size);
    StrideB* stride_B_device = (StrideB*)((uint8_t*)stride_A_device + config.stride_As_size);
    StrideC* stride_C_device = (StrideC*)((uint8_t*)stride_B_device + config.stride_Bs_size);
    StrideD* stride_D_device = (StrideD*)((uint8_t*)stride_C_device + config.stride_Cs_size);

    std::vector<ElementA*> ptr_A_host(config.gemm_groups);
    std::vector<ElementB*> ptr_B_host(config.gemm_groups);
    std::vector<ElementC*> ptr_C_host(config.gemm_groups);
    std::vector<ElementC*> ptr_D_host(config.gemm_groups);

    std::vector<StrideA> stride_A_host;
    std::vector<StrideB> stride_B_host;
    std::vector<StrideC> stride_C_host;
    std::vector<StrideD> stride_D_host;  // use stride_C_host?

    int64_t total_elements_A = 0;
    int64_t total_elements_B = 0;
    int64_t total_elements_C = 0;
    // int64_t total_elements_D = 0;

    config.host_buffer.reserve(config.problem_sizes);
    int32_t* problem_sizes_host = (int32_t*)config.host_buffer.data();
    int32_t index = 0;
    int32_t N = config.N_d;
    int32_t K = config.K;
    int64_t elements_B = K * N;
    const int64_t* expert_offset = (const int64_t*)expert_offset_host;
    for (int32_t i = 0; i < config.gemm_groups; i++, index += 3) {
        int32_t M = expert_offset[i + 1] - expert_offset[i];
        problem_sizes_host[index] = M;
        problem_sizes_host[index + 1] = N;
        problem_sizes_host[index + 2] = K;

        ptr_A_host.at(i) = (ElementA *)input + total_elements_A;
        ptr_B_host.at(i) = (ElementB *)weight + total_elements_B;
        ptr_C_host.at(i) = (ElementC *)matrix_cs + total_elements_C;
        ptr_D_host.at(i) = (ElementC *)matrix_ds + total_elements_C;

        int64_t elements_A = M * K;
        int64_t elements_C = M * N;

        total_elements_A += elements_A;
        total_elements_B += elements_B;
        total_elements_C += elements_C;
        // total_elements_D += elements_C;

        stride_A_host.push_back(cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1}));
        stride_B_host.push_back(cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1}));
        stride_C_host.push_back(cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1})); // stride<N, 1, 0>
        stride_D_host.push_back(cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1}));
    }
    int32_t index_p = 0;  //debug
    for (int32_t i = 0; i < config.gemm_groups; i++) {
        std::cout << i << ", problem size: " << problem_sizes_host[index_p] << ", " << problem_sizes_host[index_p+1] << ", " << problem_sizes_host[index_p+2] << std::endl;
        index_p += 3;
    }
    cudaMemcpy(problem_sizes_device, problem_sizes_host, config.problem_sizes, cudaMemcpyHostToDevice);
    cudaMemcpy(ptr_As_device, ptr_A_host.data(), config.ptr_x_size, cudaMemcpyHostToDevice);
    cudaMemcpy(ptr_Bs_device, ptr_B_host.data(), config.ptr_x_size, cudaMemcpyHostToDevice);
    cudaMemcpy(ptr_Cs_device, ptr_C_host.data(), config.ptr_x_size, cudaMemcpyHostToDevice);
    cudaMemcpy(ptr_Ds_device, ptr_D_host.data(), config.ptr_x_size, cudaMemcpyHostToDevice);
    cudaMemcpy(stride_A_device, stride_A_host.data(), config.stride_As_size, cudaMemcpyHostToDevice);
    cudaMemcpy(stride_B_device, stride_B_host.data(), config.stride_Bs_size, cudaMemcpyHostToDevice);
    cudaMemcpy(stride_C_device, stride_C_host.data(), config.stride_Cs_size, cudaMemcpyHostToDevice);
    cudaMemcpy(stride_D_device, stride_D_host.data(), config.stride_Ds_size, cudaMemcpyHostToDevice);

    cutlass::KernelHardwareInfo hw_info;
    // Change device_id to another value if you are running on a machine with multiple GPUs and wish
    // to use a GPU other than that with device ID 0.
    hw_info.device_id = device_id;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    // If both alpha/beta are provided (via cmd line args) and are scalar, i.e., same alpha/beta applies to all batches.
    // D = alpha * A * B + beta * C
    typename Gemm::EpilogueOutputOp::Params params;
    typename Gemm::Arguments arguments;
    if (bias != nullptr) {
        params = typename Gemm::EpilogueOutputOp::Params(
            ElementAccumulator(1.f), ElementAccumulator(1.f));
        arguments = typename Gemm::Arguments {
            cutlass::gemm::GemmUniversalMode::kGrouped,
            {config.gemm_groups, (typename ProblemShape::UnderlyingProblemShape*)problem_sizes_device, (typename ProblemShape::UnderlyingProblemShape*)problem_sizes_host},
            {ptr_As_device, stride_A_device, ptr_Bs_device, stride_B_device},
            {params, ptr_Cs_device, stride_C_device, ptr_Ds_device, stride_D_device},
            hw_info
        };
    }
    else {
        params = typename Gemm::EpilogueOutputOp::Params(
            ElementAccumulator(1.f), ElementAccumulator(0.f));
        arguments = typename Gemm::Arguments {
            cutlass::gemm::GemmUniversalMode::kGrouped,
            {config.gemm_groups, (typename ProblemShape::UnderlyingProblemShape*)problem_sizes_device, (typename ProblemShape::UnderlyingProblemShape*)problem_sizes_host},
            {ptr_As_device, stride_A_device, ptr_Bs_device, stride_B_device},
            {params, nullptr, nullptr, ptr_Ds_device, stride_D_device},
            hw_info
        };
    }

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    std::cout << "workspace size(0): " << workspace_size << std::endl;
    int32_t arguments_workspace_id = enqueue_arguments(arguments);
    config.arguments_workspace_id = arguments_workspace_id;
    config.workspace_size = workspace_size;

    std::cout << "config.arguments_workspace_id: " << config.arguments_workspace_id << std::endl;
    std::cout << "config.workspace_size: " << config.workspace_size << std::endl;
    std::cout << "hw_info.sm_count: " << hw_info.sm_count << std::endl;

    return ppl::common::RC_SUCCESS;
}

// input [seqlen * num_experts_per_token, hidden_dim]
// weight [num_experts_per_token, hidden_dim_out/w, hidden_dim]
// offset [num_experts_per_token + 1]
// gemm_output [seqlen * num_experts_per_token, hidden_dim_out/w]
// output [seqlen * num_experts_per_token, hidden_dim_out]
ppl::common::RetCode moe_column_parallel_linear2(
    const cudaStream_t stream,
    const ppl::common::NcclParam* nccl_param,
    const bool gather_output,
    void * workspace,
    moe_column_parallel_linear_config& config,
    const ppl::common::TensorShape* output_shape,
    void* output)
{
    int32_t arguments_workspace_id = config.arguments_workspace_id;
    Gemm::Arguments arguments = gemm_arguments[arguments_workspace_id].arguments;

    // Instantiate CUTLASS kernel depending on templates
    Gemm gemm;

    //std::cout << "before gemm.get_workspace_size" << std::endl;
    size_t wp_size = gemm.get_workspace_size(arguments);
    std::cout << "workspace size(1): " << wp_size << std::endl;

    // Check if the problem size is supported or not
    //std::cout << "before gemm.can_implement" << std::endl;
    CUTLASS_CHECK(gemm.can_implement(arguments));
    //std::cout << "after gemm.can_implement" << std::endl;

    // Initialize CUTLASS kernel with arguments and workspace pointer
    CUTLASS_CHECK(gemm.initialize(arguments, workspace));

    // Correctness / Warmup iteration
    CUTLASS_CHECK(gemm.run());

    remove_arguments(arguments_workspace_id);

    if (gather_output && nccl_param->size > 1) {
        ppl::common::RetCode status = ppl::common::NcclAllGather<half>(
            (half*)config.matrix_ds,
            (half*)config.gather_buffer,
            config.M * config.N_d,
            nccl_param,
            stream);
        if (ppl::common::RC_SUCCESS != status)
            return status;

        // gather_buffer(w, config.M, N/w)
        status = PPLCUDATranspose01ForwardImp(
            stream, config.gather_buffer,
            output_shape->GetDataType(),
            nccl_param->size,
            config.M,
            config.N_d,
            output);
        if (ppl::common::RC_SUCCESS != status)
            return status;
    }

    return ppl::common::RC_SUCCESS;
}

}}}}}
