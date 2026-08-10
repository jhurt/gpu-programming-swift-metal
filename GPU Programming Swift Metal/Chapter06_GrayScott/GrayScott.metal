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

struct GrayScottParams {
    float F;  // feed rate
    float K;  // kill rate
    float Du; // U diffusion coefficient
    float Dv; // V diffusion coefficient
    float dt; // time step
};

using DiffusionFunction = float2(uint cellX,
                                 uint cellY,
                                 uint cellIndex,
                                 uint2 gridSize,
                                 const device float* currentU,
                                 const device float* currentV,
                                 GrayScottParams params);

// (1) add the `[[visible]]` attribute so we can get a `MTLFunction` object of this function
[[visible]] float2 isotropicDiffusion(uint cellX,
                                      uint cellY,
                                      uint cellIndex,
                                      uint2 gridSize,
                                      const device float* currentU,
                                      const device float* currentV,
                                      GrayScottParams params) {
    float threadLocalU = currentU[cellIndex];
    float threadLocalV = currentV[cellIndex];
    
    float laplacianU = 0.0;
    float laplacianV = 0.0;
    
    // (2) use isotropic weights so all neighbors have the same weight
    constexpr float centerWeight = -1.0;
    constexpr float adjacentWeight = 0.20;
    constexpr float diagonalWeight = 0.05;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = clamp(int(cellX) + dx, 0, int(gridSize.x) - 1);
            int ny = clamp(int(cellY) + dy, 0, int(gridSize.y) - 1);
            int idx = ny * gridSize.x + nx;
            
            float weight;
            if (dx == 0 && dy == 0) {
                weight = centerWeight;
            } else if (dx == 0 || dy == 0) {
                weight = adjacentWeight;
            } else {
                weight = diagonalWeight;
            }
            
            laplacianU += currentU[idx] * weight;
            laplacianV += currentV[idx] * weight;
        }
    }
    
    // (3) Gray-Scott equations
    float uvv = threadLocalU * threadLocalV * threadLocalV;
    float du = (params.Du * laplacianU) - uvv + params.F * (1.0 - threadLocalU);
    float dv = (params.Dv * laplacianV) + uvv - (params.F + params.K) * threadLocalV;
    float2 result;
    result.x = saturate(threadLocalU + du * params.dt);
    result.y = saturate(threadLocalV + dv * params.dt);
    
    // (4) return the result
    return result;
}

// (1) add the `[[visible]]` attribute so we can get a `MTLFunction` object of this function
[[visible]] float2 anisotropicDiffusion(uint cellX,
                                        uint cellY,
                                        uint cellIndex,
                                        uint2 gridSize,
                                        const device float* currentU,
                                        const device float* currentV,
                                        GrayScottParams params) {
    float threadLocalU = currentU[cellIndex];
    float threadLocalV = currentV[cellIndex];
    
    float laplacianU = 0.0;
    float laplacianV = 0.0;
    
    // (2) use fixed anisotropic weights (≈ 0.3)
    constexpr float centerWeight = -1.0;
    constexpr float adjacentXWeight = 0.40; // boosted diffusion along X
    constexpr float adjacentYWeight = 0.05; // reduced diffusion along Y
    constexpr float diagonalWeight = 0.025;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = clamp(int(cellX) + dx, 0, int(gridSize.x) - 1);
            int ny = clamp(int(cellY) + dy, 0, int(gridSize.y) - 1);
            int idx = ny * gridSize.x + nx;
            
            float weight;
            if (dx == 0 && dy == 0) {
                weight = centerWeight;
            } else if (dy == 0) {
                weight = adjacentXWeight;
            } else if (dx == 0) {
                weight = adjacentYWeight;
            } else {
                weight = diagonalWeight;
            }
            
            laplacianU += currentU[idx] * weight;
            laplacianV += currentV[idx] * weight;
        }
    }
    
    // (3) Gray-Scott equations
    float uvv = threadLocalU * threadLocalV * threadLocalV;
    float du = (params.Du * laplacianU) - uvv + params.F * (1.0 - threadLocalU);
    float dv = (params.Dv * laplacianV) + uvv - (params.F + params.K) * threadLocalV;
    float2 result;
    result.x = saturate(threadLocalU + du * params.dt);
    result.y = saturate(threadLocalV + dv * params.dt);
    
    // (4) return the result
    return result;
}

kernel void reactionDiffusion(const device float* currentU [[buffer(0)]],
                              const device float* currentV [[buffer(1)]],
                              device float* nextU [[buffer(2)]],
                              device float* nextV [[buffer(3)]],
                              constant uint2& gridSize [[buffer(4)]],
                              constant GrayScottParams& params [[buffer(5)]],
                              uint2 threadPositionInGrid [[thread_position_in_grid]]) {
    if (threadPositionInGrid.x >= gridSize.x || threadPositionInGrid.y >= gridSize.y) {
        return;
    }
    
    int x = int(threadPositionInGrid.x);
    int y = int(threadPositionInGrid.y);
    uint cellIndex = y * gridSize.x + x;
    // (1) get currentU and currentU at the cell index for this thread
    float threadLocalU = currentU[cellIndex];
    float threadLocalV = currentV[cellIndex];
    
    // (2) approximate Laplacian 3x3 convolution with 3x3 Laplacian Stencil
    float laplacianU = 0.0;
    float laplacianV = 0.0;
    float centerWeight = -1.0;
    float adjacentWeight = 0.20;
    float diagonalWeight = 0.05;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = clamp(x + dx, 0, int(gridSize.x) - 1);
            int ny = clamp(y + dy, 0, int(gridSize.y) - 1);
            int index = ny * gridSize.x + nx;
            
            float weight;
            if (dx == 0 && dy == 0) {
                weight = centerWeight;
            } else if (dx == 0 || dy == 0) {
                weight = adjacentWeight;
            } else {
                weight = diagonalWeight;
            }
            
            laplacianU += currentU[index] * weight;
            laplacianV += currentV[index] * weight;
        }
    }
    
    // (3) Gray-Scott equations
    float uvv = threadLocalU * threadLocalV * threadLocalV;
    float du = (params.Du * laplacianU) - uvv + params.F * (1.0 - threadLocalU);
    float dv = (params.Dv * laplacianV) + uvv - (params.F + params.K) * threadLocalV;
    
    // (4) write to `nextU` and `nextV`, saturate ensures we stay between 0.0 and 1.0
    nextU[cellIndex] = saturate(threadLocalU + du * params.dt);
    nextV[cellIndex] = saturate(threadLocalV + dv * params.dt);
}

kernel void reactionDiffusionMultipleEnvironment(const device float* currentU [[buffer(0)]],
                                                 const device float* currentV [[buffer(1)]],
                                                 device float* nextU [[buffer(2)]],
                                                 device float* nextV [[buffer(3)]],
                                                 constant uint2& gridSize [[buffer(4)]],
                                                 constant GrayScottParams* environmentPool [[buffer(5)]],
                                                 constant uint& totalEnvironments [[buffer(6)]],
                                                 constant uint* cellEnvironments [[buffer(7)]],
                                                 visible_function_table<DiffusionFunction> diffusionFunctionTable [[buffer(8)]],
                                                 uint2 threadPositionInGrid [[thread_position_in_grid]]) {
    if (threadPositionInGrid.x >= gridSize.x || threadPositionInGrid.y >= gridSize.y) {
        return;
    }
    
    // (1) calculate a flattened cell index and lookup the environment at this location
    uint cellIndex = threadPositionInGrid.y * gridSize.x + threadPositionInGrid.x;
    uint environmentID = cellEnvironments[cellIndex];
    GrayScottParams params = environmentPool[environmentID];
    
    // (2) select whether to use isotropic or anisotropic diffusion based on environment
    uint functionID = 0;
    if (environmentID % 3 == 0) {
        functionID = 1;
    }
    
    // (3) get a pointer to the diffusion function at `functionID`
    DiffusionFunction* diffusionFunction = diffusionFunctionTable[functionID];
    
    // (4) execute the function and use the resultant float2 to update `nextU` and `nextV`
    float2 result = diffusionFunction(threadPositionInGrid.x, threadPositionInGrid.y, cellIndex, gridSize, currentU, currentV, params);
    nextU[cellIndex] = result.x;
    nextV[cellIndex] = result.y;
}

// (1) define the macro
#define ISOTROPIC_DIFFUSION 1
// #define ANISOTROPIC_DIFFUSION

kernel void reactionDiffusionMultipleEnvironmentPreprocessorMacro(const device float* currentU [[buffer(0)]],
                                                                  const device float* currentV [[buffer(1)]],
                                                                  device float* nextU [[buffer(2)]],
                                                                  device float* nextV [[buffer(3)]],
                                                                  constant uint2& gridSize [[buffer(4)]],
                                                                  constant GrayScottParams* environmentPool [[buffer(5)]],
                                                                  constant uint& totalEnvironments [[buffer(6)]],
                                                                  constant uint* cellEnvironments [[buffer(7)]],
                                                                  uint2 threadPositionInGrid [[thread_position_in_grid]]) {
    if (threadPositionInGrid.x >= gridSize.x || threadPositionInGrid.y >= gridSize.y) {
        return;
    }
    
    uint cellIndex = threadPositionInGrid.y * gridSize.x + threadPositionInGrid.x;
    uint environmentID = cellEnvironments[cellIndex];
    GrayScottParams params = environmentPool[environmentID];
    
    // (2) select whether to use isotropic or anisotropic diffusion based on preprocessor macro
    float2 result;
#ifdef ISOTROPIC_DIFFUSION
    result = isotropicDiffusion(threadPositionInGrid.x, threadPositionInGrid.y, cellIndex, gridSize, currentU, currentV, params);
#endif
#ifdef ANISOTROPIC_DIFFUSION
    result = anisotropicDiffusion(threadPositionInGrid.x, threadPositionInGrid.y, cellIndex, gridSize, currentU, currentV, params);
#endif
    
    nextU[cellIndex] = result.x;
    nextV[cellIndex] = result.y;
}

// (1) define the function constants
constant bool useIsotropicDiffusion [[function_constant(0)]];
constant bool useAnisotropicDiffusion [[function_constant(1)]];

kernel void reactionDiffusionMultipleEnvironmentFunctionConstants(const device float* currentU [[buffer(0)]],
                                                                  const device float* currentV [[buffer(1)]],
                                                                  device float* nextU [[buffer(2)]],
                                                                  device float* nextV [[buffer(3)]],
                                                                  constant uint2& gridSize [[buffer(4)]],
                                                                  constant GrayScottParams* environmentPool [[buffer(5)]],
                                                                  constant uint& totalEnvironments [[buffer(6)]],
                                                                  constant uint* cellEnvironments [[buffer(7)]],
                                                                  uint2 threadPositionInGrid [[thread_position_in_grid]]) {
    if (threadPositionInGrid.x >= gridSize.x || threadPositionInGrid.y >= gridSize.y) {
        return;
    }
    
    uint cellIndex = threadPositionInGrid.y * gridSize.x + threadPositionInGrid.x;
    uint environmentID = cellEnvironments[cellIndex];
    GrayScottParams params = environmentPool[environmentID];
    
    // (2) select whether to use isotropic or anisotropic diffusion based on the function constant values
    float2 result;
    if (useIsotropicDiffusion) {
        result = isotropicDiffusion(threadPositionInGrid.x, threadPositionInGrid.y, cellIndex, gridSize, currentU, currentV, params);
    } else if (useAnisotropicDiffusion) {
        result = anisotropicDiffusion(threadPositionInGrid.x, threadPositionInGrid.y, cellIndex, gridSize, currentU, currentV, params);
    }
    
    nextU[cellIndex] = result.x;
    nextV[cellIndex] = result.y;
}
