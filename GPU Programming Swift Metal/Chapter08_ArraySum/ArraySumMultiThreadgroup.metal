// Copyright 2026 Jason Hurt
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <metal_stdlib>
using namespace metal;

#include "ArraySumShared.h"

kernel void sumDeviceMemoryMultiThreadgroup(const device float* input [[buffer(0)]],
                                            constant ulong& count [[buffer(1)]],
                                            constant ulong& partitionLength [[buffer(2)]],
                                            device float* output [[buffer(3)]],
                                            uint threadPositionInGrid [[thread_position_in_grid]]) {
    ulong startIndex = threadPositionInGrid * partitionLength;
    if (startIndex >= count) {
        return;
    }
    
    ulong endIndex = min(startIndex + partitionLength, count);
    
    float threadLocalSum = kahanSummation(input, startIndex, endIndex);
    
    output[threadPositionInGrid] = threadLocalSum;
}

kernel void sumAtomicsMultiThreadgroup(const device float* input [[buffer(0)]],
                                       constant ulong& count [[buffer(1)]],
                                       device atomic_float* output [[buffer(2)]],
                                       uint threadPositionInGrid [[thread_position_in_grid]],
                                       uint threadsPerGrid [[threads_per_grid]]) {
    // (1) determine partition length based on threads per grid rather than threads per threadgroup
    ulong partitionLength = (count + threadsPerGrid - 1) / threadsPerGrid;
    ulong startIndex = threadPositionInGrid * partitionLength;
    if (startIndex >= count) {
        return;
    }
    
    ulong endIndex = min(startIndex + partitionLength, count);
    
    float threadLocalSum = kahanSummation(input, startIndex, endIndex);
    
    atomic_fetch_add_explicit(output, threadLocalSum, memory_order_relaxed);
}

kernel void sumAtomicsSIMDReduction(const device float *input [[buffer(0)]],
                                    constant ulong &count [[buffer(1)]],
                                    device atomic_float* output [[buffer(2)]],
                                    uint threadPositionInGrid [[thread_position_in_grid]],
                                    uint threadsPerGrid [[threads_per_grid]],
                                    uint threadsPerSimdgroup [[threads_per_simdgroup]],
                                    uint threadIndexInSimdgroup [[thread_index_in_simdgroup]]) {
    ulong partitionLength = (count + threadsPerGrid - 1) / threadsPerGrid;
    ulong startIndex = threadPositionInGrid * partitionLength;
    if (startIndex >= count) {
        return;
    }
    
    ulong endIndex = min(startIndex + partitionLength, count);
    
    // (1) each thread will store its partial sum into `threadLocalSum`
    float threadLocalSum = kahanSummation(input, startIndex, endIndex);
    
    // (2) SIMD reduction across each SIMD-group
    for (uint offset = threadsPerSimdgroup >> 1; offset > 0; offset >>= 1) {
        threadLocalSum += simd_shuffle_down(threadLocalSum, offset);
    }
    
    // (3) first thread in each SIMD-group increments the total sum with the partial sum for this SIMD-group
    if (threadIndexInSimdgroup == 0) {
        atomic_fetch_add_explicit(output, threadLocalSum, memory_order_relaxed);
    }
}

kernel void sumAtomicsSIMDReductionUnrolled(const device float *input [[buffer(0)]],
                                            constant ulong &count [[buffer(1)]],
                                            device atomic_float* output [[buffer(2)]],
                                            uint threadPositionInGrid [[thread_position_in_grid]],
                                            uint threadsPerGrid [[threads_per_grid]],
                                            uint threadIndexInSimdgroup [[thread_index_in_simdgroup]]) {
    ulong partitionLength = (count + threadsPerGrid - 1) / threadsPerGrid;
    ulong startIndex = threadPositionInGrid * partitionLength;
    if (startIndex >= count) {
        return;
    }
    
    ulong endIndex = min(startIndex + partitionLength, count);
    
    float threadLocalSum = kahanSummation(input, startIndex, endIndex);
    
    // (1) manually unroll SIMD reduction loop
    threadLocalSum += simd_shuffle_down(threadLocalSum, 16);
    threadLocalSum += simd_shuffle_down(threadLocalSum, 8);
    threadLocalSum += simd_shuffle_down(threadLocalSum, 4);
    threadLocalSum += simd_shuffle_down(threadLocalSum, 2);
    threadLocalSum += simd_shuffle_down(threadLocalSum, 1);
    
    // (2) first thread in each SIMD-group increments the total sum with the partial sum for this SIMD-group
    if (threadIndexInSimdgroup == 0) {
        atomic_fetch_add_explicit(output, threadLocalSum, memory_order_relaxed);
    }
}

kernel void sumAtomicsSIMDSum(const device float *input [[buffer(0)]],
                              constant ulong &count [[buffer(1)]],
                              device atomic_float* output [[buffer(2)]],
                              uint threadPositionInGrid [[thread_position_in_grid]],
                              uint threadsPerGrid [[threads_per_grid]],
                              uint threadIndexInSimdgroup [[thread_index_in_simdgroup]]) {
    ulong partitionLength = (count + threadsPerGrid - 1) / threadsPerGrid;
    ulong startIndex = threadPositionInGrid * partitionLength;
    if (startIndex >= count) {
        return;
    }
    
    ulong endIndex = min(startIndex + partitionLength, count);
    
    float threadLocalSum = kahanSummation(input, startIndex, endIndex);

    // (1) use the simd_sum function to broadcast group's partial sum to all threads in SIMD-group
    float groupSum = simd_sum(threadLocalSum);
    
    // (2) all threads in each SIMD-group have the group's partial sum, only one needs to add it to the total
    if (threadIndexInSimdgroup == 0) {
        atomic_fetch_add_explicit(output, groupSum, memory_order_relaxed);
    }
}

kernel void sumAtomicsSIMDSumPackedFloat(const device float4 *input [[buffer(0)]],
                                         constant ulong &count [[buffer(1)]],
                                         device atomic_float* output [[buffer(2)]],
                                         uint threadPositionInGrid [[thread_position_in_grid]],
                                         uint threadsPerGrid [[threads_per_grid]],
                                         uint threadIndexInSimdgroup [[thread_index_in_simdgroup]]) {
    ulong partitionLength = (count + threadsPerGrid - 1) / threadsPerGrid;
    ulong startIndex = threadPositionInGrid * partitionLength;
    if (startIndex >= count) {
        return;
    }
    
    ulong endIndex = min(startIndex + partitionLength, count);
    
    float threadLocalSum = kahanSummationFloat4(input, startIndex, endIndex);

    // (1) use the simd_sum function to broadcast group's partial sum to all threads in SIMD-group
    float groupSum = simd_sum(threadLocalSum);
    
    // (2) all threads in each SIMD-group have the group's partial sum, only one needs to add it to the total
    if (threadIndexInSimdgroup == 0) {
        atomic_fetch_add_explicit(output, groupSum, memory_order_relaxed);
    }
}
