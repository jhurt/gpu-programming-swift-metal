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

constant uint m[[function_constant(0)]];
constant uint n[[function_constant(1)]];
constant uint p[[function_constant(2)]];

kernel void systolicArrayGemm2x2Threadgroup(const device float* A [[buffer(0)]],
                                            const device float* B [[buffer(1)]],
                                            device float* C [[buffer(2)]],
                                            uint threadPositionInGrid [[thread_position_in_grid]]) {
    // (1) determine the row and column of each thread's PE
    const uint row = threadPositionInGrid / 2;
    const uint col = threadPositionInGrid % 2;
    
    // (2) declare threadgroup memory for each PE for both A and B, 4 elements wide for the 4 PEs
    threadgroup float sharedA[4];
    threadgroup float sharedB[4];
    
    float a = 0.0f;
    float b = 0.0f;
    float c = 0.0f;
    
    // cycle 0
    if (row == 0) {
        if (col == 0) {
            // (1) read from device memory A
            a = A[0];
            // (2) write to threadgroup memory A
            sharedA[threadPositionInGrid] = a;
            
            // (3) read from device memory B
            b = B[0];
            // (4) write to threadgroup memory B
            sharedB[threadPositionInGrid] = b;
            
            // (5) MAC for PE 0,0
            c += a * b;
        }
    }
    // (6) block waiting for threadgroup memory writes
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // cycle 1
    if (row == 0) {
        if (col == 0) {
            // (1) PE 0,0 reads from device memory A and B
            a = A[1];
            b = B[2];
            c += a * b;
        } else if (col == 1) {
            // (2) PE 0,1 gets `a` from its left neighbor and `b` from device memory
            a = sharedA[0];
            b = B[1];
            c += a * b;
        }
        
    } else if (row == 1) {
        if (col == 0) {
            // (3) PE 1,0 gets `a` from device memory and `b` from its top neighbor
            a = A[2];
            b = sharedB[0];
            c += a * b;
        }
    }
    // (4) block waiting for reads from cycle 1 to complete
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // (5) each thread writes its a and b from cycle 1 to threadgroup memory in preparation for cycle 2
    sharedA[threadPositionInGrid] = a;
    sharedB[threadPositionInGrid] = b;
    // (6) block waiting for threadgroup memory writes above
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // cycle 2
    if (row == 0) {
        if (col == 1) {
            // (1) PE 0,1 gets `a` from its left neighbor and `b` from device memory
            a = sharedA[0];
            b = B[3];
            c += a * b;
        }
    } else if (row == 1) {
        if (col == 0) {
            // (2) PE 1,0 gets `a` from device memory and `b` from its top neighbor
            a = A[3];
            b = sharedB[0];
            c += a * b;
        } else if (col == 1) {
            // (3) PE 1,1 gets `a` from its left neighbor and `b` from its top neighbor
            a = sharedA[2];
            b = sharedB[1];
            c += a * b;
        }
    }
    
    // (4) same block, write, block pattern as cycle 1
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sharedA[threadPositionInGrid] = a;
    sharedB[threadPositionInGrid] = b;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // cycle 3
    if (row == 1) {
        if (col == 1) {
            // (1) PE 1,1 gets `a` from its left neighbor and `b` from its top neighbor
            a = sharedA[2];
            b = sharedB[1];
            c += a * b;
        }
    }
    
    // (2) each PE writes its `c` to device memory C
    C[threadPositionInGrid] = c;
}

kernel void systolicArrayGemmThreadgroup(const device float* A [[buffer(0)]],
                                         const device float* B [[buffer(1)]],
                                         device float* C [[buffer(2)]],
                                         threadgroup float* sharedA [[threadgroup(0)]],
                                         threadgroup float* sharedB [[threadgroup(1)]],
                                         uint threadPositionInGrid [[thread_position_in_grid]]) {
    // (1) determine the row and column of each thread's PE
    const uint row = threadPositionInGrid / p;
    const uint col = threadPositionInGrid % p;
    
    float a = 0.0f;
    float b = 0.0f;
    float c = 0.0f;
    
    // (2) calculate the total number of cycles needed based on m, n, and p
    uint cycles = (m - 1) + (p - 1) + n;
    for(uint cycle = 0; cycle < cycles; cycle++) {
        if (col == 0) {
            uint c = cycle - row;
            if (c >= 0 && c < n) {
                // (3) determine where in A to retrieve `a` based on this PE's row and which cycle we are on
                a = A[row * n + c];
            } else {
                // (4) write a 0 if the PE is not performing a MAC this cycle
                a = 0.0f;
            }
        } else {
            // (5) for all PEs not in column 0, get the `a` value from the left neighbor
            a = sharedA[row * p + (col - 1)];
        }
        
        if (row == 0) {
            uint r = cycle - col;
            if (r >= 0 && r < n) {
                // (6) determine where in B to retrieve `b` based on this PE's column and which cycle we are on
                b = B[r * p + col];
            } else {
                // (7) write a 0 if the PE is not performing a MAC this cycle
                b = 0.0f;
            }
        } else {
            // (8) for all PEs not in row 0, get the `b` value from the top neighbor
            b = sharedB[(row - 1) * p + col];
        }
        
        // (9) MAC
        c += a * b;
        
        // (10) block waiting for reads from this cycle to complete
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // (11) each PE writes its `a` and `b` to threadgroup memory to pass to their neighbors on the next cycle
        sharedA[threadPositionInGrid] = a;
        sharedB[threadPositionInGrid] = b;
        
        // (12) block waiting for threadgroup memory writes above
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // (13) each PE writes its `c` to device memory C
    if (row < m && col < p) {
        C[row * p + col] = c;
    }
}

kernel void systolicArrayGemmSIMD(const device float* A [[buffer(0)]],
                                  const device float* B [[buffer(1)]],
                                  device float* C [[buffer(2)]],
                                  uint threadIndexInSimdgroup [[thread_index_in_simdgroup]]) {
    uint row = threadIndexInSimdgroup / p;
    uint col = threadIndexInSimdgroup % p;
    
    float a = 0.0f;
    float b = 0.0f;
    float c = 0.0f;
    
    uint cycles = (m - 1) + (p - 1) + n;
    for (uint cycle = 0; cycle < cycles; cycle++) {
        // (1) each PE receives its left neighbor's `a` value into leftA
        float leftA = simd_shuffle(a, threadIndexInSimdgroup - 1);
        if (col == 0) {
            uint c = cycle - row;
            if (c >= 0 && c < n) {
                a = A[row * n + c];
            } else {
                a = 0.0f;
            }
        } else {
            // (2) for all PEs not in column 0, assign the `a` value from the left neighbor
            a = leftA;
        }
        
        // (3) each PE receives its top neighbor's `b` value into topB
        float topB = simd_shuffle(b, threadIndexInSimdgroup - p);
        if (row == 0) {
            uint r = cycle - col;
            if (r >= 0 && r < n) {
                b = B[r * p + col];
            } else {
                b = 0.0f;
            }
        } else {
            // (4) for all PEs not in row 0, assign the `b` value from the top neighbor
            b = topB;
        }
        
        c += a * b;
    }
    
    if (row < m && col < p) {
        C[row * p + col] = c;
    }
}
