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

#include "FFTShared.h"

kernel void fftCooleyTukey(const device float2* inputBuffer [[buffer(0)]],
                           constant uint& N [[buffer(1)]],
                           constant float& angleMultiplier [[buffer(2)]],
                           constant float& normalizationFactor [[buffer(3)]],
                           device float2* outputBuffer [[buffer(4)]],
                           threadgroup float2* sharedRow [[threadgroup(0)]],
                           uint2 threadPositionInThreadgroup [[thread_position_in_threadgroup]],
                           uint2 threadgroupPositionInGrid [[threadgroup_position_in_grid]]) {
    // (1) determine the two natural indices of the input in the row that this threadgroup will calculate
    uint index0 = threadPositionInThreadgroup.x;
    uint index1 = index0 + (N >> 1);
    
    // (2) perform the bit reversal, reverse_bits() and the shift handles the mapping from natural to bit-reversed indices
    uint bitsNeeded = ctz(N);
    uint reverseIndex0 = reverse_bits(index0) >> (32 - bitsNeeded);
    uint reverseIndex1 = reverse_bits(index1) >> (32 - bitsNeeded);
    
    // (3) write to the sharedRow threadgroup memory the two bit reversed values from the input buffer
    uint rowOffset = threadgroupPositionInGrid.y * N;
    sharedRow[reverseIndex0] = inputBuffer[rowOffset + index0];
    sharedRow[reverseIndex1] = inputBuffer[rowOffset + index1];
    
    // (4) block waiting for all threads in the threadgroup to finish their bit reversals
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // (5) butterfly stage loop, a breadth-first iterative version of the recursive Cooley-Tukey
    for (uint stageSize = 2; stageSize <= N; stageSize <<= 1) {
        uint halfSize = stageSize >> 1;
        
        // (6) determine the angle for the twiddle factor, the angle is negative for the forward FFT and positive for the inverse FFT
        float angle = angleMultiplier * M_PI_F * (float)(index0 % halfSize) / (float)stageSize;
        
        // (7) calculate the twiddle factor
        float2 twiddle = calculateTwiddle(angle);
        
        // (8) multiply the twiddle factor by upper value in the sharedRow threadgroup memory
        uint butterflyIndex = index0 / halfSize * stageSize + index0 % halfSize;
        float2 v0 = sharedRow[butterflyIndex];
        float2 v1 = sharedRow[butterflyIndex + halfSize];
        float2 rotatedLower = complexMultiply(twiddle, v1);
        
        // (9) write the updated values to the sharedRow threadgroup memory
        sharedRow[butterflyIndex] = v0 + rotatedLower;
        sharedRow[butterflyIndex + halfSize] = v0 - rotatedLower;
        
        // (10) block waiting for all threads in this threadgroup to finish writing new values for each stage to threadgroup memory
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // (11) write the final output values to the output buffer
    if (normalizationFactor != 1.0f) {
        // (12) normalize in the final inverse FFT to get back the original magnitude
        outputBuffer[rowOffset + index0] = sharedRow[index0] * normalizationFactor;
        outputBuffer[rowOffset + index1] = sharedRow[index1] * normalizationFactor;
    } else {
        outputBuffer[rowOffset + index0] = sharedRow[index0];
        outputBuffer[rowOffset + index1] = sharedRow[index1];
    }
}

kernel void fftCooleyTukeySIMD(const device float2* inputBuffer [[buffer(0)]],
                               constant uint& N [[buffer(1)]],
                               constant float& angleMultiplier [[buffer(2)]],
                               constant float& normalizationFactor [[buffer(3)]],
                               device float2* outputBuffer [[buffer(4)]],
                               threadgroup float2* sharedRow [[threadgroup(0)]],
                               uint2 threadPositionInThreadgroup [[thread_position_in_threadgroup]],
                               uint2 threadgroupPositionInGrid [[threadgroup_position_in_grid]],
                               uint threadIndexInSimdGroup [[thread_index_in_simdgroup]]) {
    uint index0 = threadPositionInThreadgroup.x;
    uint index1 = index0 + (N >> 1);
    
    uint bitsNeeded = ctz(N);
    uint reverseIndex0 = reverse_bits(index0) >> (32 - bitsNeeded);
    uint reverseIndex1 = reverse_bits(index1) >> (32 - bitsNeeded);
    
    uint rowOffset = threadgroupPositionInGrid.y * N;
    sharedRow[reverseIndex0] = inputBuffer[rowOffset + index0];
    sharedRow[reverseIndex1] = inputBuffer[rowOffset + index1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // (1) write the two values to registers in preparation for SIMD shuffle
    float2 v0 = sharedRow[index0];
    float2 v2 = sharedRow[index1];
    float2 a;
    float2 b;
    
    // (2) SIMD butterfly stage loop (2, 4, 8, 16, 32)
    for (uint stageSize = 2; stageSize <= 32 && stageSize <= N; stageSize <<= 1) {
        uint halfSize = stageSize >> 1;
        
        // (3) get v0 and v2 partner values
        float2 v1 = simd_shuffle_xor(v0, halfSize);
        float2 v3 = simd_shuffle_xor(v2, halfSize);
        
        bool isLower = threadIndexInSimdGroup & halfSize;
        // (4) lower partner calculates twiddle factor and does multiplications
        if (isLower) {
            float angle = angleMultiplier * M_PI_F * (float)(threadIndexInSimdGroup % halfSize) / (float)stageSize;
            float2 twiddle = calculateTwiddle(angle);
            a = complexMultiply(twiddle, v0);
            b = complexMultiply(twiddle, v2);
            v0 = v1 - a;
            v2 = v3 - b;
        }
        
        // (5) upper partner receives a and b from lower partner
        a = simd_shuffle_xor(a, halfSize);
        b = simd_shuffle_xor(b, halfSize);
        if(!isLower) {
            v0 = v0 + a;
            v2 = v2 + b;
        }
    }
    
    // (6) write SIMD results to threadgroup memory
    sharedRow[index0] = v0;
    sharedRow[index1] = v2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // (7) threadgroup butterfly stages
    for (uint stageSize = 64; stageSize <= N; stageSize <<= 1) {
        uint halfSize = stageSize >> 1;
        
        float angle = angleMultiplier * M_PI_F * (float)(index0 % halfSize) / (float)stageSize;
        float2 twiddle = calculateTwiddle(angle);
        
        uint butterflyIndex = index0 / halfSize * stageSize + index0 % halfSize;
        float2 v0 = sharedRow[butterflyIndex];
        float2 v1 = sharedRow[butterflyIndex + halfSize];
        float2 rotatedLower = complexMultiply(twiddle, v1);
        
        sharedRow[butterflyIndex] = v0 + rotatedLower;
        sharedRow[butterflyIndex + halfSize] = v0 - rotatedLower;
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (normalizationFactor != 1.0f) {
        outputBuffer[rowOffset + index0] = sharedRow[index0] * normalizationFactor;
        outputBuffer[rowOffset + index1] = sharedRow[index1] * normalizationFactor;
    } else {
        outputBuffer[rowOffset + index0] = sharedRow[index0];
        outputBuffer[rowOffset + index1] = sharedRow[index1];
    }
}

kernel void fftCooleyTukeySIMDExtraMath(const device float2* inputBuffer [[buffer(0)]],
                                        constant uint& N [[buffer(1)]],
                                        constant float& angleMultiplier [[buffer(2)]],
                                        constant float& normalizationFactor [[buffer(3)]],
                                        device float2* outputBuffer [[buffer(4)]],
                                        threadgroup float2* sharedRow [[threadgroup(0)]],
                                        uint2 threadPositionInThreadgroup [[thread_position_in_threadgroup]],
                                        uint2 threadgroupPositionInGrid [[threadgroup_position_in_grid]],
                                        uint threadIndexInSimdGroup [[thread_index_in_simdgroup]]) {
    uint index0 = threadPositionInThreadgroup.x;
    uint index1 = index0 + (N >> 1);
    
    uint bitsNeeded = ctz(N);
    uint reverseIndex0 = reverse_bits(index0) >> (32 - bitsNeeded);
    uint reverseIndex1 = reverse_bits(index1) >> (32 - bitsNeeded);
    
    uint rowOffset = threadgroupPositionInGrid.y * N;
    sharedRow[reverseIndex0] = inputBuffer[rowOffset + index0];
    sharedRow[reverseIndex1] = inputBuffer[rowOffset + index1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float2 v0 = sharedRow[index0];
    float2 v2 = sharedRow[index1];
    for (uint stageSize = 2; stageSize <= 32 && stageSize <= N; stageSize <<= 1) {
        uint halfSize = stageSize >> 1;
        
        float angle = angleMultiplier * M_PI_F * (float)(threadIndexInSimdGroup % halfSize) / (float)stageSize;
        float2 twiddle = calculateTwiddle(angle);
        
        float2 v1 = simd_shuffle_xor(v0, halfSize);
        float2 v3 = simd_shuffle_xor(v2, halfSize);
        
        bool isLower = threadIndexInSimdGroup & halfSize;
        if (isLower) {
            v0 = v1 - complexMultiply(twiddle, v0);
            v2 = v3 - complexMultiply(twiddle, v2);
        } else {
            v0 = v0 + complexMultiply(twiddle, v1);
            v2 = v2 + complexMultiply(twiddle, v3);
        }
    }
    
    sharedRow[index0] = v0;
    sharedRow[index1] = v2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stageSize = 64; stageSize <= N; stageSize <<= 1) {
        uint halfSize = stageSize >> 1;
        
        float angle = angleMultiplier * M_PI_F * (float)(index0 % halfSize) / (float)stageSize;
        float2 twiddle = calculateTwiddle(angle);
        
        uint butterflyIndex = index0 / halfSize * stageSize + index0 % halfSize;
        float2 v0 = sharedRow[butterflyIndex];
        float2 v1 = sharedRow[butterflyIndex + halfSize];
        float2 rotatedLower = complexMultiply(twiddle, v1);
        
        sharedRow[butterflyIndex] = v0 + rotatedLower;
        sharedRow[butterflyIndex + halfSize] = v0 - rotatedLower;
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (normalizationFactor != 1.0f) {
        outputBuffer[rowOffset + index0] = sharedRow[index0] * normalizationFactor;
        outputBuffer[rowOffset + index1] = sharedRow[index1] * normalizationFactor;
    } else {
        outputBuffer[rowOffset + index0] = sharedRow[index0];
        outputBuffer[rowOffset + index1] = sharedRow[index1];
    }
}

kernel void fftTranspose(const device float2* inputBuffer [[buffer(0)]],
                         constant uint& width [[buffer(1)]],
                         constant uint& height [[buffer(2)]],
                         device float2* outputBuffer [[buffer(3)]],
                         uint2 threadPositionInThreadgroup [[thread_position_in_threadgroup]],
                         uint2 threadgroupPositionInGrid [[threadgroup_position_in_grid]]) {
    threadgroup float2 tile[32][32];
    
    uint x = threadgroupPositionInGrid.x * 32 + threadPositionInThreadgroup.x;
    uint y = threadgroupPositionInGrid.y * 32 + threadPositionInThreadgroup.y;
    if (x < width && y < height) {
        tile[threadPositionInThreadgroup.y][threadPositionInThreadgroup.x] = inputBuffer[y * width + x];
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    uint transposedX = threadgroupPositionInGrid.y * 32 + threadPositionInThreadgroup.x;
    uint transposedY = threadgroupPositionInGrid.x * 32 + threadPositionInThreadgroup.y;
    if (transposedX < height && transposedY < width) {
        outputBuffer[transposedY * height + transposedX] = tile[threadPositionInThreadgroup.x][threadPositionInThreadgroup.y];
    }
}
