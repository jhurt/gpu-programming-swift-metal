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

kernel void mergeSortOddEven(device int* data [[buffer(0)]],
                             constant uint& mergeSize [[buffer(1)]],
                             constant uint& stride [[buffer(2)]],
                             uint threadPositionInGrid [[thread_position_in_grid]]) {
    // (1) determine the index of one of the elements to compare and swap based on each thread's position in the grid
    uint index = 2 * threadPositionInGrid - (threadPositionInGrid & (stride - 1));

    // (2) enter the odd-indexed compare and swap if the stride is less than half the merge size
    if (stride < mergeSize >> 1) {
        // (3) find the local index of the thread within the data, equivalent to: threadPosistionInGrid % (mergeSize / 2)
        uint offset = threadPositionInGrid & ((mergeSize >> 1) - 1);
        if (offset >= stride) {
            // (4) odd-indexed compare and swap
            int a = data[index - stride];
            int b = data[index];
            if (a > b) {
                data[index - stride] = b;
                data[index] = a;
            }
        }
    } else {
        // (5) even-indexed compare and swap
        int a = data[index];
        int b = data[index + stride];
        if (a > b) {
            data[index] = b;
            data[index + stride] = a;
        }
    }
}

kernel void mergeSortOddEvenThreadgroup(device int* data [[buffer(0)]],
                                        uint threadPositionInThreadgroup [[thread_position_in_threadgroup]],
                                        uint threadgroupPositionInGrid [[threadgroup_position_in_grid]]) {
    // (1) each thread reads two values from device memory and writes them to threadgroup memory
    threadgroup int shared[1024];
    shared[threadPositionInThreadgroup] = data[threadgroupPositionInGrid * 1024 + threadPositionInThreadgroup];
    shared[threadPositionInThreadgroup + 512] = data[threadgroupPositionInGrid * 1024 + threadPositionInThreadgroup + 512];
    // (2) block waiting for all threads in the threadgroup to finish threadgroup memory writes
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // (3) combine the Swift loops from the device memory solution with the device memory kernel logic
    for(uint mergeSize = 2; mergeSize <= 1024; mergeSize <<= 1) {
        for(uint stride = mergeSize >> 1; stride > 0; stride >>= 1) {
            uint index = 2 * threadPositionInThreadgroup - (threadPositionInThreadgroup & (stride - 1));
            if (stride < mergeSize >> 1) {
                uint offset = threadPositionInThreadgroup & ((mergeSize >> 1) - 1);
                if (offset >= stride) {
                    int a = shared[index - stride];
                    int b = shared[index];
                    if (a > b) {
                        shared[index - stride] = b;
                        shared[index] = a;
                    }
                }
            } else {
                int a = shared[index];
                int b = shared[index + stride];
                if (a > b) {
                    shared[index] = b;
                    shared[index + stride] = a;
                }
            }
            
            // (4) block waiting for all threads in the threadgroup to finish their swaps for this iteration
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
        }
    }
    
    // (5) copy values from shared memory to device memory
    data[threadgroupPositionInGrid * 1024 + threadPositionInThreadgroup] = shared[threadPositionInThreadgroup];
    data[threadgroupPositionInGrid * 1024 + threadPositionInThreadgroup + 512] = shared[threadPositionInThreadgroup + 512];
}

kernel void mergeSortBitonic(device int* data [[buffer(0)]],
                             constant uint& length [[buffer(1)]],
                             constant uint& mergeSize [[buffer(2)]],
                             constant uint& stride [[buffer(3)]],
                             uint threadPositionInGrid [[thread_position_in_grid]]) {
    // (1) determine the direction of the sort of this thread based on its position in the grid
    uint direction = (threadPositionInGrid & ((length >> 1) - 1) & (mergeSize >> 1)) != 0;
    
    // (2) determine the index of one of the elements to compare and swap based on its position in the grid
    uint index = 2 * threadPositionInGrid - (threadPositionInGrid & (stride - 1));

    // (3) compare and swap device memory values
    int a = data[index];
    int b = data[index + stride];
    if ((direction == 1 && a < b) || (direction == 0 && b < a)) {
        data[index] = b;
        data[index + stride] = a;
    }
}

kernel void mergeSortBitonicThreadgroup(device int* data [[buffer(0)]],
                                        constant uint& length [[buffer(1)]],
                                        uint threadPositionInThreadgroup [[thread_position_in_threadgroup]],
                                        uint threadgroupPositionInGrid [[threadgroup_position_in_grid]]) {
    // (1) each thread reads two values from device memory and writes them to threadgroup memory
    threadgroup int shared[2048];
    shared[threadPositionInThreadgroup] = data[threadgroupPositionInGrid * 2048 + threadPositionInThreadgroup];
    shared[threadPositionInThreadgroup + 1024] = data[threadgroupPositionInGrid * 2048 + threadPositionInThreadgroup + 1024];
    // (2) block waiting for all threads in the threadgroup to finish threadgroup memory writes
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // (3) combine the Swift loops from the device memory solution with the device memory kernel logic
    for(uint mergeSize = 2; mergeSize <= 1024; mergeSize <<= 1) {
        for(uint stride = mergeSize >> 1; stride > 0; stride >>= 1) {
            uint direction = (threadPositionInThreadgroup & ((length >> 1) - 1) & (mergeSize >> 1)) != 0;
            uint index = 2 * threadPositionInThreadgroup - (threadPositionInThreadgroup & (stride - 1));
            int a = shared[index];
            int b = shared[index + stride];
            if ((direction == 1 && a < b) || (direction == 0 && b < a)) {
                shared[index] = b;
                shared[index + stride] = a;
            }
            
            // (4) block waiting for all threads in the threadgroup to finish their swaps for this iteration
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    
    // (5) copy values from shared memory to device memory
    data[threadgroupPositionInGrid * 2048 + threadPositionInThreadgroup] = shared[threadPositionInThreadgroup];
    data[threadgroupPositionInGrid * 2048 + threadPositionInThreadgroup + 1024] = shared[threadPositionInThreadgroup + 1024];
}
