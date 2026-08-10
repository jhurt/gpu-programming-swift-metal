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

// (1) function constant for the loop count
constant int forLoopIterations [[function_constant(0)]];

// (2) a kernel function without any unroll directive
kernel void regularForLoop(device float* buffer [[buffer(0)]],
                           uint threadPositionInGrid [[thread_position_in_grid]]) {
    float result = buffer[threadPositionInGrid];
    for(int i = 0; i < forLoopIterations; i++) {
        result += sin(result) * 2.0f;
    }
    buffer[threadPositionInGrid] = result;
}

// (3) a kernel function with the unroll(disable) directive
kernel void divergentForLoop(device float* buffer [[buffer(0)]],
                             uint threadPositionInGrid [[thread_position_in_grid]]) {
    float result = buffer[threadPositionInGrid];
#pragma clang loop unroll(disable)
    for(int i = 0; i < forLoopIterations; i++) {
        result += sin(result) * 2.0f;
    }
    buffer[threadPositionInGrid] = result;
}

// (4) a kernel function the the unroll(full) directive
kernel void unrolledForLoop(device float* buffer [[buffer(0)]],
                            uint threadPositionInGrid [[thread_position_in_grid]]) {
    float result = buffer[threadPositionInGrid];
#pragma clang loop unroll(full)
    for(int i = 0; i < forLoopIterations; i++) {
        result += sin(result) * 2.0f;
    }
    buffer[threadPositionInGrid] = result;
}
