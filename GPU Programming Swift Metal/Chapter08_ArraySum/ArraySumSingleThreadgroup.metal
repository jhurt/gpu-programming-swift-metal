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

kernel void sumDeviceMemoryNumericalErrors(const device float* input [[buffer(0)]],
                                           constant ulong& count [[buffer(1)]],
                                           device float* output [[buffer(2)]],
                                           uint threadIndexInThreadgroup [[thread_index_in_threadgroup]],
                                           uint threadsPerThreadgroup [[threads_per_threadgroup]]) {
    // (1) determine partition offsets for this thread
    ulong partitionLength = (count + threadsPerThreadgroup - 1) / threadsPerThreadgroup;
    ulong startIndex = threadIndexInThreadgroup * partitionLength;
    if (startIndex >= count) {
        return;
    }
    ulong endIndex = min(startIndex + partitionLength, count);
    
    // (2) sum this thread's partition
    float threadLocalSum = 0.0;
    for (ulong i = startIndex; i < endIndex; i++) {
        threadLocalSum += input[i];
    }
    
    // (3) write the result to device memory
    output[threadIndexInThreadgroup] = threadLocalSum;
    
    // (4) ensure all threads in the threadgroup have written their local sums
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // (5) first thread of each threadgroup sums the output buffer entries
    if (threadIndexInThreadgroup == 0) {
        threadLocalSum = 0.0f;
        for (uint i = 0; i < threadsPerThreadgroup; i++) {
            threadLocalSum += output[i];
        }
        
        // (6) write global sum to beginning of output buffer
        output[0] = threadLocalSum;
    }
}

kernel void sumDeviceMemory(const device float* input [[buffer(0)]],
                            constant ulong& count [[buffer(1)]],
                            device float* output [[buffer(2)]],
                            uint threadIndexInThreadgroup [[thread_index_in_threadgroup]],
                            uint threadsPerThreadgroup [[threads_per_threadgroup]]) {
    // (1) determine partition offsets for this thread
    ulong partitionLength = (count + threadsPerThreadgroup - 1) / threadsPerThreadgroup;
    ulong startIndex = threadIndexInThreadgroup * partitionLength;
    if (startIndex >= count) {
        return;
    }
    ulong endIndex = min(startIndex + partitionLength, count);
    
    // (2) sum this thread's partition
    float threadLocalSum = kahanSummation(input, startIndex, endIndex);
    
    // (3) write the result to device output buffer
    output[threadIndexInThreadgroup] = threadLocalSum;
    
    // (4) ensure all threads in the threadgroup have written their local sums
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // (5) first thread sums the output buffer entries
    if (threadIndexInThreadgroup == 0) {
        threadLocalSum = 0.0f;
        for (uint i = 0; i < threadsPerThreadgroup; i++) {
            threadLocalSum += output[i];
        }
        
        // (6) write global sum to beginning of output buffer
        output[0] = threadLocalSum;
    }
}

kernel void sumThreadgroupMemory(const device float* input [[buffer(0)]],
                                 constant ulong& count [[buffer(1)]],
                                 device float* output [[buffer(2)]],
                                 threadgroup float* sharedMemory [[threadgroup(0)]],
                                 uint threadIndexInThreadgroup [[thread_index_in_threadgroup]],
                                 uint threadsPerThreadgroup [[threads_per_threadgroup]]) {
    ulong partitionLength = (count + threadsPerThreadgroup - 1) / threadsPerThreadgroup;
    ulong startIndex = threadIndexInThreadgroup * partitionLength;
    if (startIndex >= count) {
        return;
    }
    ulong endIndex = min(startIndex + partitionLength, count);
    
    float threadLocalSum = kahanSummation(input, startIndex, endIndex);
    
    // (1) write the result to shared threadgroup memory
    sharedMemory[threadIndexInThreadgroup] = threadLocalSum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (threadIndexInThreadgroup == 0) {
        threadLocalSum = 0.0f;
        for (uint i = 0; i < threadsPerThreadgroup; i++) {
            // (2) partial sums are read from shared threadgroup memory
            threadLocalSum += sharedMemory[i];
        }
        
        output[0] = threadLocalSum;
    }
}

kernel void sumThreadgroupMemoryGridStride(const device float* input [[buffer(0)]],
                                           constant ulong& count [[buffer(1)]],
                                           device float* output [[buffer(2)]],
                                           threadgroup float* sharedMemory [[threadgroup(0)]],
                                           uint threadIndexInThreadgroup [[thread_index_in_threadgroup]],
                                           uint threadsPerThreadgroup [[threads_per_threadgroup]]) {
    // (1) grid-stride to get the partial sum for this thread
    float threadLocalSum = kahanSummationStride(input, count, threadIndexInThreadgroup, threadsPerThreadgroup);
    
    sharedMemory[threadIndexInThreadgroup] = threadLocalSum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (threadIndexInThreadgroup == 0) {
        threadLocalSum = 0.0f;
        for (uint i = 0; i < threadsPerThreadgroup; i++) {
            threadLocalSum += sharedMemory[i];
        }
        
        output[0] = threadLocalSum;
    }
}

kernel void sumAtomics(const device float* input [[buffer(0)]],
                       constant ulong& count [[buffer(1)]],
                       device atomic_float* output [[buffer(2)]], // (1) use atomic_float for the output buffer
                       uint threadIndexInThreadgroup [[thread_index_in_threadgroup]],
                       uint threadsPerThreadgroup [[threads_per_threadgroup]]) {
    ulong partitionLength = (count + threadsPerThreadgroup - 1) / threadsPerThreadgroup;
    ulong startIndex = threadIndexInThreadgroup * partitionLength;
    if (startIndex >= count) {
        return;
    }
    ulong endIndex = min(startIndex + partitionLength, count);
    
    float threadLocalSum = kahanSummation(input, startIndex, endIndex);
    
    // (2) each thread increments the output by its partial sum
    atomic_fetch_add_explicit(output, threadLocalSum, memory_order_relaxed);
}
