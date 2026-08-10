//
//  FFTShared.h
//  ParallelSwift
//
//  Created by Jason Hurt on 2/1/26.
//

#ifndef FFTShared_h
#define FFTShared_h

#include <metal_stdlib>
using namespace metal;

inline float2 complexMultiply(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline float2 calculateTwiddle(float angle) {
    float c, s;
    s = sincos(angle, c);
    return float2(c, s);
}

#endif /* FFTShared_h */
