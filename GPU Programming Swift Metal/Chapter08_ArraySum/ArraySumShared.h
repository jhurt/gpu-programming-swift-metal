//
//  ArraySumShared.h
//  ParallelSwift
//
//  Created by Jason Hurt on 8/3/26.
//

#ifndef ArraySumShared_h
#define ArraySumShared_h

#include <metal_stdlib>
using namespace metal;

inline float kahanSummation(const device float* buffer, ulong startIndex, ulong endIndex) {
#pragma METAL fp math_mode(safe) // (1) set math_mode option to safe which disables unsafe floating-point optimizations
    float sum = 0.0f; // (2) accumulator
    float c = 0.0f; // (3) running compensation for lost low-order bits
    for (ulong i = startIndex; i < endIndex; i++) {
        // (4) c is zero the first time around
        float y = buffer[i] - c;
        // (5) sum is big, y small, so low-order digits of y are lost
        float t = sum + y;
        // (6) (t - sum) cancels the high-order part of y, subtracting y recovers negative (low part of y)
        c = (t - sum) - y;
        sum = t;
    }
    
    return sum;
}

inline float kahanSummationFloat4(const device float4* buffer, ulong startIndex, ulong endIndex) {
#pragma METAL fp math_mode(safe) // (1) set math_mode option to safe which disables unsafe floating-point optimizations
    float4 sum = float4(0.0f); // (2) accumulator vector
    float4 c = float4(0.0f);  // (3) running compensation for lost low-order bits
    for (ulong i = startIndex; i < endIndex; i++) {
        float4 v = buffer[i];
        // (4) c is zero the first time around
        float4 y = v - c;
        // (5) sum is big, y small, so low-order digits of y are lost
        float4 t = sum + y;
        // (6) (t - sum) cancels the high-order part of y, subtracting y recovers negative (low part of y)
        c = (t - sum) - y;
        sum = t;
    }
    
    return sum.x + sum.y + sum.z + sum.w;
}

inline float kahanSummationStride(const device float* buffer, ulong count, uint startIndex, uint stride) {
#pragma METAL fp math_mode(safe)
    float sum = 0.0f; // accumulator
    float c = 0.0f; // running compensation for lost low-order bits
    for (ulong i = startIndex; i < count; i += stride) {
        float y = buffer[i] - c; // c is zero the first time around
        // sum is big, y small, so low-order digits of y are lost
        float t = sum + y;
        // (t - sum) cancels the high-order part of y; subtracting y recovers negative (low part of y)
        c = (t - sum) - y;
        sum = t;
    }
    
    return sum;
}

#endif /* ArraySumShared_h */
