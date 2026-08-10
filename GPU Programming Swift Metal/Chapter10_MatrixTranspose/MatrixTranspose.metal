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

kernel void matrixTranspose(const device float* input [[buffer(0)]],
                            constant uint& columns [[buffer(1)]],
                            constant uint& rows [[buffer(2)]],
                            constant uint& tileSize [[buffer(3)]],
                            device float* output [[buffer(4)]],
                            threadgroup float* tile [[threadgroup(0)]],
                            uint2 threadPositionInThreadgroup [[thread_position_in_threadgroup]],
                            uint2 threadgroupPositionInGrid [[threadgroup_position_in_grid]]) {
    // (1) determine the coordinates in the input matrix each thread is responsible for
    uint x = threadgroupPositionInGrid.x * tileSize + threadPositionInThreadgroup.x;
    uint y = threadgroupPositionInGrid.y * tileSize + threadPositionInThreadgroup.y;
    
    // (2) read from the device memory input and write to threadgroup memory tile
    tile[threadPositionInThreadgroup.y * tileSize + threadPositionInThreadgroup.x] = input[y * columns + x];
    
    // (3) block waiting for all threads in threadgroup to finish their writes
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // (4) determine the transposed coordinates in the output matrix that each thread is responsible for
    uint transposedX = threadgroupPositionInGrid.y * tileSize + threadPositionInThreadgroup.x;
    uint transposedY = threadgroupPositionInGrid.x * tileSize + threadPositionInThreadgroup.y;
    
    // (5) each thread will write a single value from tile threadgroup memory to device output memory
    output[transposedY * rows + transposedX] = tile[threadPositionInThreadgroup.x * tileSize + threadPositionInThreadgroup.y];
}
