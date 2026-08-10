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

kernel void fftStockham(const device float2* inputBuffer [[buffer(0)]],
                        device float2* outputBuffer [[buffer(1)]],
                        constant uint& iteration [[buffer(2)]],
                        constant uint& N [[buffer(3)]],
                        constant float& angleMultiplier [[buffer(4)]],
                        constant float& normalizationFactor [[buffer(5)]],
                        uint2 threadPositionInGrid [[thread_position_in_grid]]) {
    // (1) calculate the two read indices for each thread based on the stage and thread position in the grid
    uint stride = 1 << iteration;
    uint block = threadPositionInGrid.x / stride;
    uint offset = threadPositionInGrid.x % stride;
    uint readIndex0 = block * stride + offset;
    uint readIndex1 = readIndex0 + (N >> 1);
    
    // (2) read two values from the input
    uint rowOffset = threadPositionInGrid.y * N;
    float2 v0 = inputBuffer[rowOffset + readIndex0];
    float2 v1 = inputBuffer[rowOffset + readIndex1];
    
    // (3) calculate the twiddle factor and multiply it by the lower value from device memory
    float angle = angleMultiplier * M_PI_F * (float)offset / (float)(2 * stride);
    float2 twiddle = calculateTwiddle(angle);
    float2 v2 = complexMultiply(twiddle, v1);

    // (4) write the updated values to device memory
    uint writeIndex0 = block * (2 * stride) + offset;
    uint writeIndex1 = writeIndex0 + stride;
    if (normalizationFactor != 1.0f) {
        outputBuffer[rowOffset + writeIndex0] = normalizationFactor * (v0 + v2);
        outputBuffer[rowOffset + writeIndex1] = normalizationFactor * (v0 - v2);
    } else {
        outputBuffer[rowOffset + writeIndex0] = v0 + v2;
        outputBuffer[rowOffset + writeIndex1] = v0 - v2;
    }
}

kernel void fftStockhamRadix4(const device float2* inputBuffer [[buffer(0)]],
                              device float2* outputBuffer [[buffer(1)]],
                              constant uint& iteration [[buffer(2)]],
                              constant uint& N [[buffer(3)]],
                              constant float& angleMultiplier [[buffer(4)]],
                              constant float& normalizationFactor [[buffer(5)]],
                              uint2 threadPositionInGrid [[thread_position_in_grid]]) {
    // (1) calculate the four read indices for each thread based on the stage and thread position in the grid
    uint stride = 1 << (2 * iteration);
    uint block = threadPositionInGrid.x / stride;
    uint offset = threadPositionInGrid.x % stride;
    uint readIndex0 = block * stride + offset;
    uint w = N >> 2;
    uint readIndex1 = readIndex0 + w;
    uint readIndex2 = readIndex1 + w;
    uint readIndex3 = readIndex2 + w;
    
    // (2) read four values from the input
    uint rowOffset = threadPositionInGrid.y * N;
    float2 value0 = inputBuffer[rowOffset + readIndex0];
    float2 value1 = inputBuffer[rowOffset + readIndex1];
    float2 value2 = inputBuffer[rowOffset + readIndex2];
    float2 value3 = inputBuffer[rowOffset + readIndex3];
    
    // (3) calculate multiple twiddles factor and rotate multiple values read from device memory
    float angle = angleMultiplier * M_PI_F * (float)offset / (float)(4 * stride);
    float2 twiddle1 = calculateTwiddle(angle);
    float2 twiddle2 = calculateTwiddle(angle * 2.0f);
    float2 twiddle3 = calculateTwiddle(angle * 3.0f);
    float2 t0 = value0;
    float2 t1 = complexMultiply(twiddle1, value1);
    float2 t2 = complexMultiply(twiddle2, value2);
    float2 t3 = complexMultiply(twiddle3, value3);
    
    // (4) radix-4 butterfly logic
    float2 a = t0 + t2;
    float2 b = t0 - t2;
    float2 c = t1 + t3;
    float2 d;
    if (angleMultiplier < 0.0) {
        d = float2(t1.y - t3.y, t3.x - t1.x); // (t1 - t3) * -i
    } else {
        d = float2(t3.y - t1.y, t1.x - t3.x); // (t1 - t3) * +i
    }
    
    // (5) write update values to output buffer
    uint writeIndex0 = block * (4 * stride) + offset;
    uint writeIndex1 = writeIndex0 + stride;
    uint writeIndex2 = writeIndex1 + stride;
    uint writeIndex3 = writeIndex2 + stride;
    if (normalizationFactor != 1.0f) {
        outputBuffer[rowOffset + writeIndex0] = normalizationFactor * (a + c);
        outputBuffer[rowOffset + writeIndex1] = normalizationFactor * (b + d);
        outputBuffer[rowOffset + writeIndex2] = normalizationFactor * (a - c);
        outputBuffer[rowOffset + writeIndex3] = normalizationFactor * (b - d);
    } else {
        outputBuffer[rowOffset + writeIndex0] = a + c;
        outputBuffer[rowOffset + writeIndex1] = b + d;
        outputBuffer[rowOffset + writeIndex2] = a - c;
        outputBuffer[rowOffset + writeIndex3] = b - d;
    }
}
