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

kernel void softmax2D(const device float* attentionScores [[buffer(0)]],
                      device float* attentionWeights [[buffer(1)]],
                      constant uint& sequenceLength [[buffer(2)]],
                      constant float& attentionScale [[buffer(3)]],
                      uint threadgroupPositionInGrid [[threadgroup_position_in_grid]],
                      uint threadIndexInThreadgroup [[thread_index_in_threadgroup]]) {
    // (1) each SIMD-group handles one row of the attentionScores matrix, one SIMD-group per threadgroup
    const device float* scores = attentionScores + threadgroupPositionInGrid * sequenceLength;
    
    // (2) each thread in the SIMD-group finds a thread-local maximum
    float threadLocalMax = -INFINITY;
    for (uint i = threadIndexInThreadgroup; i < sequenceLength; i += 32) {
        threadLocalMax = max(threadLocalMax, scores[i] * attentionScale);
    }
    
    // (3) row max is the maximum of all thread-local values of all threads in each SIMD-group
    float rowMax = simd_max(threadLocalMax);
    
    // (4) each thread in the SIMD-group computes a thread-local sum of exponentials
    float threadLocalSum = 0.0f;
    for (uint i = threadIndexInThreadgroup; i < sequenceLength; i += 32) {
        threadLocalSum += fast::exp(scores[i] * attentionScale - rowMax);
    }
    
    // (5) sum all thread-local sums for each SIMD-group
    float rowExponentialSum = simd_sum(threadLocalSum);
    
    // (6) calculate the numerically stable softmax value for each row and write it to the output buffer
    device float* weights = attentionWeights + (threadgroupPositionInGrid * sequenceLength);
    if (rowExponentialSum > 1e-9f) {
        for (uint i = threadIndexInThreadgroup; i < sequenceLength; i += 32) {
            weights[i] = fast::exp(scores[i] * attentionScale - rowMax) / rowExponentialSum;
        }
    } else {
        for (uint i = threadIndexInThreadgroup; i < sequenceLength; i += 32) {
            weights[i] = 0.0f;
        }
    }
}
