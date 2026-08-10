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
#include <metal_tensor>
using namespace metal;

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// from MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h
kernel void matrixMultiplyTensor(tensor<device float, dextents<int32_t, 2>> A [[buffer(0)]],
                                 tensor<device float, dextents<int32_t, 2>> B [[buffer(1)]],
                                 tensor<device float, dextents<int32_t, 2>> C [[buffer(2)]],
                                 uint2 threadgroupPositionInGrid [[threadgroup_position_in_grid]]) {
    // (1) define the parameters for 2D tiled GEMM
    constexpr auto matmulDescriptor = matmul2d_descriptor(64, // m outer dim of local tile
                                                          32, // n outer dim of local tile
                                                          static_cast<int>(dynamic_extent), // k inner dimension. dynamic_extent means operation will read k from input tensor
                                                          false, // transpose_left
                                                          false, // transpose_right
                                                          true, // relaxed_precision
                                                          matmul2d_descriptor::mode::multiply);
    
    // (2) create a GEMM operator that will execute with 4 SIMD-groups
    matmul2d<matmulDescriptor, execution_simdgroups<4>> matmulOp;
    
    // (3) each threadgroup will calculate a 64x32 chunk of C
    auto sliceA = A.slice(0, threadgroupPositionInGrid.y * 64);
    auto sliceB = B.slice(threadgroupPositionInGrid.x * 32, 0);
    auto sliceC = C.slice(threadgroupPositionInGrid.x * 32, threadgroupPositionInGrid.y * 64);

    // (4) execute the operation
    matmulOp.run(sliceA, sliceB, sliceC);
}

kernel void reshapeAndTransposeQKVTensor(tensor<device float, dextents<int32_t, 2>> input [[buffer(0)]], // (embeddingDimension, sequenceLength)
                                         tensor<device float, dextents<int32_t, 3>> output [[buffer(1)]], // (headDimension, sequenceLength, headCount)
                                         constant uint& sequenceLength [[buffer(2)]],
                                         constant uint& headCount [[buffer(3)]],
                                         constant uint& headDimension [[buffer(4)]],
                                         uint3 threadPositionInGrid [[thread_position_in_grid]]) {
    const uint tokenIndex = threadPositionInGrid.x; // 0..sequenceLength - 1
    const uint headIndex = threadPositionInGrid.y; // 0..headCount - 1
    const uint dimensionIndex = threadPositionInGrid.z; // 0..headDimension - 1

    output[dimensionIndex, tokenIndex, headIndex] = input[headIndex * headDimension + dimensionIndex, tokenIndex];
}

kernel void multiHeadScoresTensor(tensor<device float, dextents<int32_t, 3>> queries [[buffer(0)]], // (headDimension, sequenceLength, headCount)
                                  tensor<device float, dextents<int32_t, 3>> keys [[buffer(1)]], // (headDimension, sequenceLength, headCount)
                                  tensor<device float, dextents<int32_t, 3>> attentionScores [[buffer(2)]], // (sequenceLength, sequenceLength, headCount)
                                  constant uint& headDimension [[buffer(3)]],
                                  constant float& attentionScale [[buffer(4)]],
                                  uint3 threadPositionInGrid [[thread_position_in_grid]]) {
    const uint queryRow = threadPositionInGrid.x;
    const uint keyRow = threadPositionInGrid.y;
    const uint headIndex = threadPositionInGrid.z;
    
    float sum = 0.0f;
    for (uint i = 0; i < headDimension; i++) {
        sum += queries[i, queryRow, headIndex] * keys[i, keyRow, headIndex];
    }
    
    attentionScores[keyRow, queryRow, headIndex] = sum * attentionScale;
}

kernel void softmax3DTensor(tensor<device float, dextents<int32_t, 3>> attentionScores [[buffer(0)]], // (sequenceLength, sequenceLength, headCount)
                            tensor<device float, dextents<int32_t, 3>> attentionWeights [[buffer(1)]], // (sequenceLength, sequenceLength, headCount)
                            constant uint& sequenceLength [[buffer(2)]],
                            uint3 threadPositionInGrid [[thread_position_in_grid]],
                            uint threadIndexInThreadgroup [[thread_index_in_threadgroup]]) {
    const uint headIndex = threadPositionInGrid.z;
    const uint row = threadPositionInGrid.y;
    
    float threadLocalMax = -INFINITY;
    for (uint i = threadIndexInThreadgroup; i < sequenceLength; i += 32) {
        threadLocalMax = max(threadLocalMax, attentionScores[i, row, headIndex]);
    }
    float rowMax = simd_max(threadLocalMax);
    
    float threadLocalSum = 0.0f;
    for (uint i = threadIndexInThreadgroup; i < sequenceLength; i += 32) {
        threadLocalSum += exp((attentionScores[i, row, headIndex]) - rowMax);
    }
    float rowSum = simd_sum(threadLocalSum);
    
    if (rowSum > 1e-9f) {
        for (uint i = threadIndexInThreadgroup; i < sequenceLength; i += 32) {
            attentionWeights[i, row, headIndex] = fast::exp(attentionScores[i, row, headIndex] - rowMax) / rowSum;
        }
    } else {
        for (uint i = threadIndexInThreadgroup; i < sequenceLength; i += 32) {
            attentionWeights[i, row, headIndex] = 0.0f;
        }
    }
}

kernel void multiHeadContextVectorsTensor(tensor<device float, dextents<int32_t, 3>> attentionWeights [[buffer(0)]], // (sequenceLength, sequenceLength, headCount)
                                          tensor<device float, dextents<int32_t, 3>> values [[buffer(1)]], // (headDimension, sequenceLength, headCount)
                                          tensor<device float, dextents<int32_t, 3>> contextVectors [[buffer(2)]], // (headDimension, sequenceLength, headCount)
                                          constant uint& sequenceLength [[buffer(3)]],
                                          constant uint& headDimension [[buffer(4)]],
                                          uint3 threadPositionInGrid [[thread_position_in_grid]]) {
    const uint headDimensionIndex = threadPositionInGrid.x;
    const uint valueRow = threadPositionInGrid.y;
    const uint headIndex = threadPositionInGrid.z;
    
    float weightedSum = 0.0f;
    for (uint i = 0; i < sequenceLength; i++) {
        weightedSum += attentionWeights[i, valueRow, headIndex] * values[headDimensionIndex, i, headIndex];
    }
    
    contextVectors[headDimensionIndex, valueRow, headIndex] = weightedSum;
}

kernel void transposeAndFlattenContextVectorsTensor(tensor<device float, dextents<int32_t, 3>> input [[buffer(0)]], // (headDimension, sequenceLength, headCount)
                                                    tensor<device float, dextents<int32_t, 2>> output [[buffer(1)]], // (embeddingDimension, sequenceLength)
                                                    constant uint& sequenceLength [[buffer(2)]],
                                                    constant uint& headCount [[buffer(3)]],
                                                    constant uint& headDimension [[buffer(4)]],
                                                    uint3 threadPositionInGrid [[thread_position_in_grid]]) {
    const uint tokenIndex = threadPositionInGrid.x; // 0..sequenceLength - 1
    const uint headIndex = threadPositionInGrid.y; // 0..headCount - 1
    const uint headDimensionIndex = threadPositionInGrid.z; // 0..headDimension - 1
    
    output[headIndex * headDimension + headDimensionIndex, tokenIndex] = input[headDimensionIndex, tokenIndex, headIndex];
}
