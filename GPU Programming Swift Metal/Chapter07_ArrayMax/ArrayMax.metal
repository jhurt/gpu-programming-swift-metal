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

kernel void maxThreadgroupMemory(const device float* input [[buffer(0)]],
                                 constant ulong& count [[buffer(1)]],
                                 constant uint& elementsPerThreadgroup [[buffer(2)]],
                                 constant uint& elementsPerThread [[buffer(3)]],
                                 device float* output [[buffer(4)]],
                                 threadgroup float* sharedMemory [[threadgroup(0)]],
                                 uint threadIndexInThreadgroup [[thread_index_in_threadgroup]],
                                 uint threadgroupPositionInGrid [[threadgroup_position_in_grid]],
                                 uint threadsPerThreadgroup [[threads_per_threadgroup]]) {
    // (1) the start index for each thread depends on the threadgroup position in the grid and the thread's position in the threadgroup
    ulong startIndex = threadgroupPositionInGrid * elementsPerThreadgroup + threadIndexInThreadgroup * elementsPerThread;
    // (2) each threadgroup processes elementsPerThread elements
    ulong endIndex = min(startIndex + elementsPerThread, count);
    
    // (3) each thread loops through its portion of the array and writes its maximum value to threadgroup memory
    float threadLocalMax = -INFINITY;
    for(uint i = 0; i < endIndex - startIndex; i++) {
        threadLocalMax = max(threadLocalMax, input[startIndex + i]);
    }
    sharedMemory[threadIndexInThreadgroup] = threadLocalMax;
    
    // (4) block waiting for all threads in the threadgroup to complete
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // (5) the first thread in each threadgroup will read find the maximum value in threadgroup memory
    if (threadIndexInThreadgroup == 0) {
        float sharedMax = -INFINITY;
        for(uint i = 0; i < threadsPerThreadgroup; i++) {
            sharedMax = max(sharedMax, sharedMemory[i]);
        }
        output[threadgroupPositionInGrid] = sharedMax;
    }
}

kernel void maxSingleSimdgroup(const device float* input [[buffer(0)]],
                               constant ulong& count [[buffer(1)]],
                               constant uint& elementsPerThreadgroup [[buffer(2)]],
                               device float* output [[buffer(3)]],
                               uint threadgroupPositionInGrid [[threadgroup_position_in_grid]],
                               uint threadIndexInSimdGroup [[thread_index_in_simdgroup]]) {
    ulong startIndex = threadgroupPositionInGrid * elementsPerThreadgroup + threadIndexInSimdGroup;
    ulong endIndex = startIndex + elementsPerThreadgroup;
    
    float threadgroupMax = -INFINITY;
    // (1) iterate through the partition by SIMD-groups per threadgroup elements
    for(ulong i = startIndex; i < endIndex; i += 32) {
        float value;
        if (i >= count) {
            value = -INFINITY;
        } else {
            value = input[i];
        }
        // (2) uses the `simd_max` function to find the maximum of all thread's `threadgroupMax` value per SIMD-group
        threadgroupMax = max(simd_max(value), threadgroupMax);
    }
    
    // (3) the first thread in each SIMD-group will write its threadgroupMax value to the output
    if(threadIndexInSimdGroup == 0) {
        output[threadgroupPositionInGrid] = threadgroupMax;
    }
}

kernel void maxMultipleSimdgroups(const device float* input [[buffer(0)]],
                                  constant ulong& count [[buffer(1)]],
                                  constant uint& elementsPerThreadgroup [[buffer(2)]],
                                  device float* output [[buffer(3)]],
                                  uint threadgroupPositionInGrid [[threadgroup_position_in_grid]],
                                  uint threadIndexInSimdGroup [[thread_index_in_simdgroup]],
                                  uint simdgroupIndexInThreadgroup [[simdgroup_index_in_threadgroup]],
                                  uint simdgroupsPerThreadgroup [[simdgroups_per_threadgroup]]) {
    uint elementsPerSimdgroup = elementsPerThreadgroup / simdgroupsPerThreadgroup;
    uint startIndex = threadgroupPositionInGrid * elementsPerThreadgroup + (simdgroupIndexInThreadgroup * elementsPerSimdgroup) + threadIndexInSimdGroup;
    ulong endIndex = startIndex + elementsPerSimdgroup;
    
    float simdgroupMax = -INFINITY;
    for(ulong i = startIndex; i < endIndex; i += 32) {
        float value;
        if (i >= count) {
            value = -INFINITY;
        } else {
            value = input[i];
        }
        simdgroupMax = max(simd_max(value), simdgroupMax);
    }
    
    if(threadIndexInSimdGroup == 0) {
        output[threadgroupPositionInGrid * simdgroupsPerThreadgroup + simdgroupIndexInThreadgroup] = simdgroupMax;
    }
}
