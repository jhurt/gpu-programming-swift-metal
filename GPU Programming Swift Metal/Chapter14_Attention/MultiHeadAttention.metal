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

kernel void reshapeAndTransposeQKV(const device float* input [[buffer(0)]], // (sequenceLength, embeddingDimension)
                                   device float* output [[buffer(1)]], // (headCount, sequenceLength, headDimension)
                                   constant uint& sequenceLength [[buffer(2)]],
                                   constant uint& headCount [[buffer(3)]],
                                   constant uint& headDimension [[buffer(4)]],
                                   uint3 threadPositionInGrid [[thread_position_in_grid]]) {
    const uint tokenIndex = threadPositionInGrid.x; // 0..sequenceLength - 1
    const uint headIndex = threadPositionInGrid.y; // 0..headCount - 1
    const uint dimensionIndex = threadPositionInGrid.z; // 0..headDimension - 1

    const uint inputIndex = tokenIndex * headCount * headDimension + headIndex * headDimension + dimensionIndex;
    const uint outputIndex = headIndex * sequenceLength * headDimension + tokenIndex * headDimension + dimensionIndex;

    output[outputIndex] = input[inputIndex];
}

kernel void multiHeadScores(const device float* queries [[buffer(0)]], // (headCount, sequenceLength, headDimension)
                            const device float* keys [[buffer(1)]], // (headCount, sequenceLength, headDimension)
                            device float* attentionScores [[buffer(2)]], // (headCount, sequenceLength, sequenceLength)
                            constant uint& sequenceLength [[buffer(3)]],
                            constant uint& headDimension [[buffer(4)]],
                            constant float& attentionScale [[buffer(5)]],
                            uint3 threadPositionInGrid [[thread_position_in_grid]]) {
    // (1) determine query row, key row, and head index
    const uint queryRow = threadPositionInGrid.x; // 0..sequenceLength - 1
    const uint keyRow = threadPositionInGrid.y; // 0..sequenceLength - 1
    const uint headIndex = threadPositionInGrid.z; // 0..headCount - 1
    
    // (2) calculate the dot product of Q_queryRow and K_keyRow for head at headIndex
    float sum = 0.0f;
    const device float* q = queries + headIndex * sequenceLength * headDimension + queryRow * headDimension;
    const device float* k = keys + headIndex * sequenceLength * headDimension + keyRow * headDimension;
    for (uint i = 0; i < headDimension; i++) {
        sum += q[i] * k[i];
    }
    
    // (3) write the dot product result to the attention scores matrix (headIndex, queryRow, keyRow)
    const uint scoreIndex = headIndex * sequenceLength * sequenceLength + queryRow * sequenceLength + keyRow;
    attentionScores[scoreIndex] = sum * attentionScale;
}

kernel void softmax3D(const device float* attentionScores [[buffer(0)]], // (headCount, sequenceLength, sequenceLength)
                      device float* attentionWeights [[buffer(1)]], // (headCount, sequenceLength, sequenceLength)
                      constant uint& sequenceLength [[buffer(2)]],
                      uint3 threadPositionInGrid [[thread_position_in_grid]],
                      uint threadIndexInThreadgroup [[thread_index_in_threadgroup]]) {
    // (1) each SIMD-group handles one row of one head of the attentionScores matrix, one SIMD-group per threadgroup
    const device float* scores = attentionScores + threadPositionInGrid.z * sequenceLength * sequenceLength + threadPositionInGrid.y * sequenceLength;
    
    float threadLocalMax = -INFINITY;
    for (uint i = threadIndexInThreadgroup; i < sequenceLength; i += 32) {
        threadLocalMax = max(threadLocalMax, scores[i]);
    }
    float rowMax = simd_max(threadLocalMax);
    
    float threadLocalSum = 0.0f;
    for (uint i = threadIndexInThreadgroup; i < sequenceLength; i += 32) {
        threadLocalSum += exp((scores[i]) - rowMax);
    }
    float rowSum = simd_sum(threadLocalSum);
    
    device float* weights = attentionWeights + threadPositionInGrid.z * sequenceLength * sequenceLength + threadPositionInGrid.y * sequenceLength;
    if (rowSum > 1e-9f) {
        for (uint i = threadIndexInThreadgroup; i < sequenceLength; i += 32) {
            weights[i] = fast::exp(scores[i] - rowMax) / rowSum;
        }
    } else {
        for (uint i = threadIndexInThreadgroup; i < sequenceLength; i += 32) {
            weights[i] = 0.0f;
        }
    }
}

kernel void multiHeadContextVectors(const device float* attentionWeights [[buffer(0)]], // (headCount, sequenceLength, sequenceLength)
                                    const device float* values [[buffer(1)]], // (headCount, sequenceLength, headDimension)
                                    device float* contextVectors [[buffer(2)]], // (headCount, sequenceLength, headDimension)
                                    constant uint& sequenceLength [[buffer(3)]],
                                    constant uint& headDimension [[buffer(4)]],
                                    uint3 threadPositionInGrid [[thread_position_in_grid]]) {
    // (1) read the head dimension, value row, and head index for each thread's weighted sum
    const uint headDimensionIndex = threadPositionInGrid.x; // 0..headDimension - 1
    const uint valueRow = threadPositionInGrid.y; // 0..sequenceLength - 1
    const uint headIndex = threadPositionInGrid.z; // 0..headCount - 1

    // (2) offset into the attention weights matrix
    const device float* weightsRow = attentionWeights + (headIndex * sequenceLength * sequenceLength) + (valueRow * sequenceLength);

    // (3) point to offset into the values matrix slice for this head
    const device float* headValues = values + headIndex * sequenceLength * headDimension;

    // (4) compute weighted sum
    float weightedSum = 0.0f;
    for (uint i = 0; i < sequenceLength; i++) {
        weightedSum += weightsRow[i] * headValues[i * headDimension + headDimensionIndex];
    }

    // (5) write the sum to the context vectors matrix
    const uint outputIndex = headIndex * sequenceLength * headDimension + (valueRow * headDimension) + headDimensionIndex;
    contextVectors[outputIndex] = weightedSum;
}

kernel void transposeAndFlattenContextVectors(const device float* input [[buffer(0)]], // (headCount, sequenceLength, headDimension)
                                              device float* output [[buffer(1)]], // (sequenceLength, embeddingDimension)
                                              constant uint& sequenceLength [[buffer(2)]],
                                              constant uint& headCount [[buffer(3)]],
                                              constant uint& headDimension [[buffer(4)]],
                                              uint3 threadPositionInGrid [[thread_position_in_grid]]) {
    const uint tokenIndex = threadPositionInGrid.x; // 0..sequenceLength - 1
    const uint headIndex = threadPositionInGrid.y; // 0..headCount - 1
    const uint headDimensionIndex = threadPositionInGrid.z; // 0..headDimension - 1
    
    const uint inputIndex = headIndex * sequenceLength * headDimension + tokenIndex * headDimension + headDimensionIndex;
    const uint outputIndex = tokenIndex * headCount * headDimension + headIndex * headDimension + headDimensionIndex;
    
    output[outputIndex] = input[inputIndex];
}

kernel void multiHeadAttentionOnlineSoftmax(const device float* queries [[buffer(0)]], // (headCount, sequenceLength, headDimension)
                                            const device float* keys [[buffer(1)]], // (headCount, sequenceLength, headDimension)
                                            const device float* values [[buffer(2)]], // (headCount, sequenceLength, headDimension)
                                            device float* contextVectors [[buffer(3)]], // (headCount, sequenceLength, headDimension)
                                            constant uint& sequenceLength [[buffer(4)]],
                                            constant uint& headCount [[buffer(5)]],
                                            constant uint& headDimension [[buffer(6)]],
                                            constant float& attentionScale [[buffer(7)]],
                                            uint3 threadPositionInGrid [[thread_position_in_grid]],
                                            uint threadIndexInSimdgroup [[thread_index_in_simdgroup]]) {
    // (1) read the query row and head index for each thread's weighted sum
    const uint queryRow = threadPositionInGrid.y;
    const uint headIndex = threadPositionInGrid.z;
    
    // (2) each thread will accumulate headDimension / 32 elements of V
    float threadLocalAccumulator[4] = {0.0f};
    const uint elementsPerThread = (headDimension + 31) / 32;
    
    float rowMax = -INFINITY;
    float rowSum = 0.0f;
    // (3) iterate through Q and K accumulating V into threadLocalAccumulator
    for (uint keyRow = 0; keyRow < sequenceLength; keyRow++) {
        // (4) partial Q * K^T
        float sum = 0.0f;
        for (uint i = threadIndexInSimdgroup; i < headDimension; i += 32) {
            sum += queries[headIndex * sequenceLength * headDimension + queryRow * headDimension + i] * keys[headIndex * sequenceLength * headDimension + keyRow * headDimension + i];
        }
        float score = simd_sum(sum) * attentionScale;
        
        // (5) online softmax
        float previousMax = rowMax;
        rowMax = max(previousMax, score);
        float expCurrent = fast::exp(score - rowMax);
        float scalePrev = fast::exp(previousMax - rowMax);
        rowSum = rowSum * scalePrev + expCurrent;
        
        // (6) accumulate weighted V
        for (uint i = 0; i < elementsPerThread; i++) {
            float value = values[headIndex * sequenceLength * headDimension + keyRow * headDimension + threadIndexInSimdgroup + (i * 32)];
            threadLocalAccumulator[i] = threadLocalAccumulator[i] * scalePrev + expCurrent * value;
        }
    }

    // (7) finalize attention output, normalize with total softmax denominator
    for (uint i = 0; i < elementsPerThread; i++) {
        const uint outputIndex = headIndex * sequenceLength * headDimension + queryRow * headDimension + threadIndexInSimdgroup + (i * 32);
        contextVectors[outputIndex] = threadLocalAccumulator[i] / rowSum;
    }
}
