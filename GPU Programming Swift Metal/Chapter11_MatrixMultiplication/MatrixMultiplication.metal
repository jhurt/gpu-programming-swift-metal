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

kernel void matrixMultiply2DGrid(const device float* A [[buffer(0)]], // m x n
                                 const device float* B [[buffer(1)]], // n x p
                                 device float* C [[buffer(2)]], // m x p
                                 uint2 threadPositionInGrid [[thread_position_in_grid]]) {
    // (1) determine the row from A and column from B based on the thread's position in the 2D grid
    uint row = threadPositionInGrid.x;
    uint col = threadPositionInGrid.y;
    
    float c = 0;
    // (2) accumulate the dot product of A[row] and B[col] into c
    for (uint k = 0; k < n; k++) {
        float a = A[row * n + k];
        float b = B[k * p + col];
        c += a * b;
    }
    
    // (3) write c to the output matrix C
    C[row * p + col] = c;
}

kernel void matrixMultiply2DGridStride(const device float* A [[buffer(0)]],
                                       const device float* B [[buffer(1)]],
                                       device float* C [[buffer(2)]],
                                       uint2 threadPositionInGrid [[thread_position_in_grid]],
                                       uint2 threadsPerGrid [[threads_per_grid]]) {
    // (1) 2D grid-stride through rows of A and columns of B
    for (uint row = threadPositionInGrid.x; row < m; row += threadsPerGrid.y) {
        for (uint col = threadPositionInGrid.y; col < p; col += threadsPerGrid.x) {
            float c = 0;
            for (uint k = 0; k < n; k++) {
                float a = A[row * n + k];
                float b = B[k * p + col];
                c += a * b;
            }
            
            C[row * p + col] = c;
        }
    }
}

kernel void matrixMultiply3DGrid(const device float* A [[buffer(0)]], // m x n
                                 const device float* B [[buffer(1)]], // n x p
                                 device atomic_float* C [[buffer(2)]], // m x p
                                 uint3 threadPositionInGrid [[thread_position_in_grid]]) {
    // (1) determine the row from A, column from B, and dot product index k based on the thread's position in the 3D grid
    int row = threadPositionInGrid.x;
    int col = threadPositionInGrid.y;
    int k = threadPositionInGrid.z;
    
    // (2) compute a single multiplication in the dot product of A and B
    float a = A[row * n + k];
    float b = B[k * p + col];
    float c = a * b;
    
    // (3) add the result to C via atomic add
    int index = row * p + col;
    atomic_fetch_add_explicit(C + index, c, memory_order_relaxed);
}

#define TILE_SIZE 16

kernel void matrixMultiplyTiledThreadgroupMemory(const device float* A [[buffer(0)]],
                                                 const device float* B [[buffer(1)]],
                                                 device float* C [[buffer(2)]],
                                                 uint2 threadPositionInGrid [[thread_position_in_grid]],
                                                 uint2 threadsPerThreadgroup [[threads_per_threadgroup]],
                                                 uint2 threadPositionInThreadgroup [[thread_position_in_threadgroup]],
                                                 uint2 threadgroupPositionInGrid [[threadgroup_position_in_grid]]) {
    // (1) declare threadgroup memory for the tiles
    threadgroup float tileA[TILE_SIZE][TILE_SIZE];
    threadgroup float tileB[TILE_SIZE][TILE_SIZE];
    
    float threadLocalSum = 0;
    // (2) loop through each tile, accumulating the tiled dot products into threadLocalSum
    for (uint tileIndex = 0; tileIndex < ceil(float(n) / TILE_SIZE); tileIndex++) {
        uint rowTileA = threadgroupPositionInGrid.y * TILE_SIZE + threadPositionInThreadgroup.y;
        uint colTileA = tileIndex * TILE_SIZE + threadPositionInThreadgroup.x;
        // (3) copy a single value from A to threadgroup memory tileA, write a 0 if we are out of bounds
        if (rowTileA < m && colTileA < n) {
            tileA[threadPositionInThreadgroup.y][threadPositionInThreadgroup.x] = A[rowTileA * n + colTileA];
        } else {
            tileA[threadPositionInThreadgroup.y][threadPositionInThreadgroup.x] = 0;
        }
        
        uint rowTileB = tileIndex * TILE_SIZE + threadPositionInThreadgroup.y;
        uint colTileB = threadgroupPositionInGrid.x * TILE_SIZE + threadPositionInThreadgroup.x;
        // (4) copy a single value from B to threadgroup memory tileB, write a 0 if we are out of bounds
        if (rowTileB < n && colTileB < p) {
            tileB[threadPositionInThreadgroup.y][threadPositionInThreadgroup.x] = B[rowTileB * p + colTileB];
        } else {
            tileB[threadPositionInThreadgroup.y][threadPositionInThreadgroup.x] = 0;
        }
        
        // (5) block here waiting for all TILE_SIZE x TILE_SIZE threads in the threadgroup to copy their values
        // from device memory to threadgroup memory
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // (6) calculates the dot product of a row from tileA and a column from tileB, reading from threadgroup memory
#pragma clang loop unroll(full)
        for (uint k = 0; k < TILE_SIZE; k++) {
            threadLocalSum += tileA[threadPositionInThreadgroup.y][k] * tileB[k][threadPositionInThreadgroup.x];
        }
        
        // (7) wait for all threads in this threadgroup to finish so that tileA and tileB can be reused for the next iteration
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // (8) write the dot product to device memory C at row * p + col
    uint row = threadgroupPositionInGrid.y * TILE_SIZE + threadPositionInThreadgroup.y;
    uint col = threadgroupPositionInGrid.x * TILE_SIZE + threadPositionInThreadgroup.x;
    if (row < m && col < p) {
        C[row * p + col] = threadLocalSum;
    }
}

kernel void matrixMultiplySimdSingleThreadgroup(const device float* A [[buffer(0)]],
                                                const device float* B [[buffer(1)]],
                                                device float* C [[buffer(2)]]) {
    // (1) iterate 8 rows at a time
    for (uint row = 0; row < m; row += 8) {
        // (2) iterate 8 columns at a time
        for(uint col = 0; col < p; col += 8) {
            // (3) declare an 8x8 accumulator c, stored in 64 registers across 32 threads in the threadgroup
            simdgroup_float8x8 c = simdgroup_float8x8(0);
            // (4) slide across shared dimension n, incrementing in 8-wide chunks
            for (uint k = 0; k < n; k += 8) {
                simdgroup_float8x8 tileA;
                simdgroup_float8x8 tileB;
                // (5) copy an 8x8 chunk from A to tileA
                simdgroup_load(tileA, A + row * n + k, n);
                // (6) copy an 8x8 chunk from B to tileB
                simdgroup_load(tileB, B + k * p + col, p);
                // (7) compute dot products for the two 8x8 chunks of A and B, accumulating with previous values in c
                simdgroup_multiply_accumulate(c, tileA, tileB, c);
            }
            
            // (8) copy c from SIMD-group registers to C index [row, col]
            simdgroup_store(c, C + row * p + col, p);
        }
    }
}

kernel void matrixMultiplySimdMultiThreadgroup1(const device float* A [[buffer(0)]],
                                                const device float* B [[buffer(1)]],
                                                device float* C [[buffer(2)]],
                                                uint2 threadgroupPositionInGrid [[threadgroup_position_in_grid]]) {
    // (1) determine row and column for this SIMD-group based on threadgroup position in the grid, one SIMD-group per threadgroup
    uint row = threadgroupPositionInGrid.y * 8;
    uint col = threadgroupPositionInGrid.x * 8;
    simdgroup_float8x8 c = simdgroup_float8x8(0);
    for (uint k = 0; k < n; k += 8) {
        simdgroup_float8x8 tileA;
        simdgroup_float8x8 tileB;
        simdgroup_load(tileA, A + row * n + k, n);
        simdgroup_load(tileB, B + k * p + col, p);
        simdgroup_multiply_accumulate(c, tileA, tileB, c);
    }
    
    simdgroup_store(c, C + row * p + col, p);
}

kernel void matrixMultiplySimdMultiThreadgroup2(const device float* A [[buffer(0)]],
                                                const device float* B [[buffer(1)]],
                                                device float* C [[buffer(2)]],
                                                uint2 threadgroupPositionInGrid [[threadgroup_position_in_grid]],
                                                uint simdgroupIndexInThreadgroup [[simdgroup_index_in_threadgroup]],
                                                uint2 threadsPerThreadgroup [[threads_per_threadgroup]]) {
    // (1) determine column for this SIMD-group based on threadgroup position in the grid, determine row based on threadgroup position and simdgroup position in its threadgroup
    uint row = threadgroupPositionInGrid.y * 8 * threadsPerThreadgroup.y + (simdgroupIndexInThreadgroup * 8);
    uint col = threadgroupPositionInGrid.x * 8;
    simdgroup_float8x8 c = simdgroup_float8x8(0);
    for (uint k = 0; k < n; k += 8) {
        simdgroup_float8x8 tileA;
        simdgroup_float8x8 tileB;
        simdgroup_load(tileA, A + row * n + k, n);
        simdgroup_load(tileB, B + k * p + col, p);
        simdgroup_multiply_accumulate(c, tileA, tileB, c);
    }
    
    simdgroup_store(c, C + row * p + col, p);
}

kernel void matrixMultiplySimd4x4(const device float* A [[buffer(0)]], // m x n
                                  const device float* B [[buffer(1)]], // n x p
                                  device float* C [[buffer(2)]], // m x p
                                  uint2 threadgroupPositionInGrid [[threadgroup_position_in_grid]],
                                  uint simdgroupIndexInThreadgroup [[simdgroup_index_in_threadgroup]],
                                  uint2 threadsPerThreadgroup [[threads_per_threadgroup]]) {
    // (1) initialize 4x4 grid of 8x8 accumulators, a 32x32 tile in C
    simdgroup_float8x8 c[4][4];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 4; i++) {
#pragma clang loop unroll(full)
        for (uint j = 0; j < 4; j++) {
            c[i][j] = simdgroup_float8x8(0);
        }
    }
    
    // (2) start row is based on threadgroup position in grid and SIMD-group position in threadgroup
    uint startRow = threadgroupPositionInGrid.y * 32 * threadsPerThreadgroup.y + (simdgroupIndexInThreadgroup * 32);
    // (3) start column is based only on threadgroup position in grid
    uint startCol = threadgroupPositionInGrid.x * 32;

    // (4) loop over the shared dimension n, incrementing by 8, consuming data in 8x8 blocks
    for (uint k = 0; k < n; k += 8) {
#pragma clang loop unroll(full)
        for (int col = 0; col < 4; col++) {
            simdgroup_float8x8 simdB;
            // (5) load 8x8 block from device memory B
            simdgroup_load(simdB, B + k * p + (startCol + col * 8), p);
#pragma clang loop unroll(full)
            for (int row = 0; row < 4; row++) {
                simdgroup_float8x8 simdA;
                // (6) load 8x8 block from device memory A
                simdgroup_load(simdA, A + (startRow + row * 8) * n + k, n);
                // (7) compute and store dot products for 8x8 block into thread registers c
                simdgroup_multiply_accumulate(c[row][col], simdA, simdB, c[row][col]);
            }
        }
        threadgroup_barrier(mem_flags::mem_none);
    }
    
#pragma clang loop unroll(full)
    for (int row = 0; row < 4; row++) {
#pragma clang loop unroll(full)
        for (int col = 0; col < 4; col++) {
            // (8) copy dot products from thread registers c into device memory C
            simdgroup_store(c[row][col], C + (startRow + row * 8) * p + (startCol + col * 8), p);
        }
    }
}

kernel void matrixMultiplySimd4x4ThreadgroupMemoryB(const device float* A [[buffer(0)]], // m x n
                                                    const device float* B [[buffer(1)]], // n x p
                                                    device float* C [[buffer(2)]], // m x p
                                                    uint2 threadgroupPositionInGrid [[threadgroup_position_in_grid]],
                                                    uint simdgroupIndexInThreadgroup [[simdgroup_index_in_threadgroup]],
                                                    uint2 threadsPerThreadgroup [[threads_per_threadgroup]]) {
    // (1) initialize 4x4 grid of 8x8 accumulators, a 32x32 tile in C
    simdgroup_float8x8 c[4][4];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 4; i++) {
#pragma clang loop unroll(full)
        for (uint j = 0; j < 4; j++) {
            c[i][j] = simdgroup_float8x8(0);
        }
    }
    
    // (2) start row is based on threadgroup position in grid and SIMD-group position in threadgroup
    uint startRow = threadgroupPositionInGrid.y * 32 * threadsPerThreadgroup.y + (simdgroupIndexInThreadgroup * 32);
    // (3) start column is based only on threadgroup position in grid
    uint startCol = threadgroupPositionInGrid.x * 32;
    
    // (4) copy an 8x32 block of B into threadgroup memory, each of the 4 SIMD-groups copies an 8x8 tile from device memory B
    threadgroup float tileB[8][32];
    simdgroup_float8x8 simdB;
    simdgroup_load(simdB, B + (startCol + simdgroupIndexInThreadgroup * 8), p);
    simdgroup_store(simdB, &tileB[0][simdgroupIndexInThreadgroup * 8], 32);
    
    for (uint k = 0; k < n; k += 8) {
        // (5) block waiting for all SIMD-groups to finish copying into tileA
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // (6) load a 32x8 block from device memory A into simdA
        simdgroup_float8x8 simdA[4];
#pragma clang loop unroll(full)
        for (uint row = 0; row < 4; row++) {
            simdgroup_load(simdA[row], A + (startRow + row * 8) * n + k, n);
        }
        
        // (7) calculate the dot products of a 32x32 tile in C
        simdgroup_float8x8 simdBB;
#pragma clang loop unroll(full)
        for (uint col = 0; col < 4; col++) {
            // (8) load an 8x8 block of threadgroup memory tileB into simdBB
            simdgroup_load(simdBB, &tileB[0][col * 8], 32);
#pragma clang loop unroll(full)
            for (uint row = 0; row < 4; row++) {
                simdgroup_multiply_accumulate(c[row][col], simdA[row], simdBB, c[row][col]);
            }
        }
        
        // (9) block so we can write to tileB again safely
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // (10) copy an 8x32 block of B into threadgroup memory, each of the 4 SIMD-groups loads an 8x8 tile from device memory B
        simdgroup_load(simdB, B + ((k+8) * p) + (startCol + simdgroupIndexInThreadgroup * 8), p);
        simdgroup_store(simdB, &tileB[0][simdgroupIndexInThreadgroup * 8], 32);
    }
    
#pragma clang loop unroll(full)
    for (uint row = 0; row < 4; row++) {
#pragma clang loop unroll(full)
        for (uint col = 0; col < 4; col++) {
            // (11) copy dot products from thread registers c into device memory C
            simdgroup_store(c[row][col], C + (startRow + row * 8) * p + (startCol + col * 8), p);
        }
    }
}

kernel void matrixMultiplySimd4x4ThreadgroupMemoryA(const device float* A [[buffer(0)]], // m x n
                                                    const device float* B [[buffer(1)]], // n x p
                                                    device float* C [[buffer(2)]], // m x p
                                                    uint2 threadgroupPositionInGrid [[threadgroup_position_in_grid]],
                                                    uint simdgroupIndexInThreadgroup [[simdgroup_index_in_threadgroup]],
                                                    uint2 threadsPerThreadgroup [[threads_per_threadgroup]]) {
    // (1) initialize 4x4 grid of 8x8 accumulators, a 32x32 tile in C
    simdgroup_float8x8 c[4][4];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 4; i++) {
#pragma clang loop unroll(full)
        for (uint j = 0; j < 4; j++) {
            c[i][j] = simdgroup_float8x8(0);
        }
    }
    
    // (2) start row is based only on threadgroup position in grid
    uint startRow = threadgroupPositionInGrid.y * 32;
    // (3) start column is based on threadgroup position in grid and SIMD-group position in threadgroup
    uint startCol = threadgroupPositionInGrid.x * 32 * threadsPerThreadgroup.y + (simdgroupIndexInThreadgroup * 32);
    
    // (4) copy a 32x8 block of A into threadgroup memory, each of the 4 SIMD-groups copies an 8x8 tile from device memory A
    threadgroup float tileA[32][8];
    simdgroup_float8x8 simdA;
    simdgroup_load(simdA, A + (startRow + simdgroupIndexInThreadgroup * 8) * n, n);
    simdgroup_store(simdA, &tileA[simdgroupIndexInThreadgroup * 8][0], 8);
    
    for (uint k = 0; k < n; k += 8) {
        // (5) block waiting for all SIMD-groups to finish copying into tileA
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // (6) load an 8x32 block from device memory B into simdB
        simdgroup_float8x8 simdB[4];
#pragma clang loop unroll(full)
        for (uint col = 0; col < 4; col++) {
            simdgroup_load(simdB[col], B + k * p + (startCol + col * 8), p);
        }
        
        // (7) calculate the dot products of a 32x32 tile in C
        simdgroup_float8x8 simdAA;
#pragma clang loop unroll(full)
        for (uint row = 0; row < 4; row++) {
            // (8) load an 8x8 block of threadgroup memory tileA into simdAA
            simdgroup_load(simdAA, &tileA[row * 8][0], 8);
#pragma clang loop unroll(full)
            for (uint col = 0; col < 4; col++) {
                simdgroup_multiply_accumulate(c[row][col], simdAA, simdB[col], c[row][col]);
            }
        }
        // (9) block so we can write to tileA again safely
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // (10) copy a 32x8 block of A into threadgroup memory, each of the 4 SIMD-groups copies an 8x8 tile from device memory A
        simdgroup_load(simdA, A + (startRow + simdgroupIndexInThreadgroup * 8) * n + (k+8), n);
        simdgroup_store(simdA, &tileA[simdgroupIndexInThreadgroup * 8][0], 8);
    }
    
#pragma clang loop unroll(full)
    for (uint row = 0; row < 4; row++) {
#pragma clang loop unroll(full)
        for (uint col = 0; col < 4; col++) {
            // (11) copy dot products from thread registers c into device memory C
            simdgroup_store(c[row][col], C + (startRow + row * 8) * p + (startCol + col * 8), p);
        }
    }
}
