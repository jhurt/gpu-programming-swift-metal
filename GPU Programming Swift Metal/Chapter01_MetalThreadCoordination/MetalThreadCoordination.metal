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

kernel void sumThreadgroupBarrier(const device int* input [[buffer(0)]],
                                  device int* output [[buffer(1)]],
                                  uint threadIndexInThreadgroup [[thread_index_in_threadgroup]],
                                  uint threadgroupPositionInGrid [[threadgroup_position_in_grid]],
                                  uint threadPositionInGrid [[thread_position_in_grid]]) {
    // (1) declare shared threadgroup memory
    threadgroup int shared[256];
    
    // (2) each thread loads one value from global to shared memory
    shared[threadIndexInThreadgroup] = input[threadPositionInGrid];
    
    // (3) use a threadgroup barrier with `mem_threadgroup` to ensure all threads in this threadgroup
    // have written their values to shared memory
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (threadIndexInThreadgroup == 0) {
        for (uint i = 0; i < 256; i++) {
            shared[0] += shared[i];
        }
        output[threadgroupPositionInGrid] = shared[0];
    }
}
