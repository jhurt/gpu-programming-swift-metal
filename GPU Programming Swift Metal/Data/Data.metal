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

ulong hashPrng(ulong seed) {
    seed = (seed ^ (seed >> 30)) * 0xbf58476d1ce4e5b9;
    seed = (seed ^ (seed >> 27)) * 0x94d049bb133111eb;
    seed = seed ^ (seed >> 31);
    return seed;
}

ulong hashPrng2(ulong seed) {
    seed = (~seed) + (seed << 21); // seed = (seed << 21) - seed - 1;
    seed = seed ^ (seed >> 24);
    seed = (seed + (seed << 3)) + (seed << 8); // seed * 265
    seed = seed ^ (seed >> 14);
    seed = (seed + (seed << 2)) + (seed << 4); // seed * 21
    seed = seed ^ (seed >> 28);
    seed = seed + (seed << 31);
    return seed;
}

kernel void randomFloat(device float* output [[buffer(0)]],
                        constant ulong& count [[buffer(1)]],
                        uint threadPositionInGrid [[thread_position_in_grid]],
                        uint threadsPerGrid [[threads_per_grid]]) {
    for (ulong i = threadPositionInGrid; i < count; i += threadsPerGrid) {
        ulong randomInt = hashPrng(i + 1);
        float randomValue = (float)randomInt / (float)0xFFFFFFFF;
        output[i] = randomValue;
    }
}

kernel void randomBoundedFloat(device float* output [[buffer(0)]],
                               constant ulong& count [[buffer(1)]],
                               constant float& lowerBound [[buffer(2)]],
                               constant float& upperBound [[buffer(3)]],
                               uint threadPositionInGrid [[thread_position_in_grid]],
                               uint threadsPerGrid [[threads_per_grid]]) {
    for (ulong i = threadPositionInGrid; i < count; i += threadsPerGrid) {
        ulong randomInt = hashPrng(i + 1);
        uint r = (uint)(randomInt >> 32);
        float randomValue = (float)r / 4294967295.0f;
        output[i] = lowerBound + (randomValue * (upperBound - lowerBound));
    }
}

kernel void randomInt(device int* output [[buffer(0)]],
                      constant ulong& count [[buffer(1)]],
                      uint threadPositionInGrid [[thread_position_in_grid]],
                      uint threadsPerGrid [[threads_per_grid]]) {
    for (ulong i = threadPositionInGrid; i < count; i += threadsPerGrid) {
        ulong randomInt = hashPrng(i + 1);
        output[i] = (int)((randomInt ^ (randomInt >> 32)) & 0xFFFFFFFF);
    }
}

kernel void floatArrayWithValues(device float* output [[buffer(0)]],
                                 constant ulong& count [[buffer(1)]],
                                 constant float& value [[buffer(2)]],
                                 uint threadPositionInGrid [[thread_position_in_grid]],
                                 uint threadsPerGrid [[threads_per_grid]]) {
    for (ulong i = threadPositionInGrid; i < count; i += threadsPerGrid) {
        output[i] = value;
    }
}

kernel void randomPoint2D(device float2* output [[buffer(0)]],
                          constant uint& count [[buffer(1)]],
                          constant int& minX [[buffer(2)]],
                          constant int& maxX [[buffer(3)]],
                          constant int& minY [[buffer(4)]],
                          constant int& maxY [[buffer(5)]],
                          uint threadIndexInThreadgroup [[thread_index_in_threadgroup]],
                          uint threadPositionInGrid [[thread_position_in_grid]]) {
    if (threadPositionInGrid < count) {
        ulong randomInt = hashPrng(threadPositionInGrid);
        float randomValue = (float)randomInt / (float)0xFFFFFFFF;
        float x = (float)minX + randomValue * (float)(maxX - minX);
        
        randomInt = hashPrng(-threadPositionInGrid);
        randomValue = (float)randomInt / (float)0xFFFFFFFF;
        float y = (float)minY + randomValue * (float)(maxY - minY);
        
        output[threadPositionInGrid] = float2(x, y);
    }
}

kernel void newULong(device ulong* output [[buffer(0)]],
                     constant ulong& count [[buffer(1)]],
                     uint threadPositionInGrid [[thread_position_in_grid]],
                     uint threadsPerGrid [[threads_per_grid]]) {
    for (ulong i = threadPositionInGrid; i < count; i += threadsPerGrid) {
        output[i] = i;
    }
}

kernel void compareFloatArrays(const device float* A [[buffer(0)]],
                               const device float* B [[buffer(1)]],
                               constant ulong& count [[buffer(2)]],
                               device atomic_uint* numberOfDiffs [[buffer(3)]],
                               device atomic_float* largestDiff [[buffer(4)]],
                               uint threadPositionInGrid [[thread_position_in_grid]],
                               uint threadsPerGrid [[threads_per_grid]]) {
    for (ulong i = threadPositionInGrid; i < count; i += threadsPerGrid) {
        if (A[i] != B[i]) {
            atomic_fetch_add_explicit(numberOfDiffs, 1, memory_order_relaxed);
            
            float diff = abs(A[i] - B[i]);
            float currentMax = atomic_load_explicit(largestDiff, memory_order_relaxed);
            if (diff > currentMax) {
                atomic_compare_exchange_weak_explicit(largestDiff, &currentMax, diff, memory_order_relaxed, memory_order_relaxed);
            }
        }
    }
}
