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

kernel void fftVisualizeFrequencyMagnitude(const device float2* fftBuffer [[buffer(0)]],
                                           texture2d<float, access::write> outputTexture [[texture(0)]],
                                           uint2 threadPositionInGrid [[thread_position_in_grid]]) {
    // calculate the shifted coordinates to bring DC to the center
    uint shiftedX = (threadPositionInGrid.x + outputTexture.get_width() / 2) % outputTexture.get_width();
    uint shiftedY = (threadPositionInGrid.y + outputTexture.get_height() / 2) % outputTexture.get_height();
    
    float2 complexVal = fftBuffer[shiftedY * outputTexture.get_width() + shiftedX];
    float magnitude = length(complexVal);
    
    // log scale
    float scaled = log(1.0f + magnitude);
    
    // normalize
    float visualizationFactor = 0.1f;
    float color = scaled * visualizationFactor;
    
    outputTexture.write(float4(float3(color), 1.0f), threadPositionInGrid);
}

kernel void fftVisualizeSpatialDomain(device const float2* ifftBuffer [[buffer(0)]],
                          texture2d<float, access::write> outputTexture [[texture(0)]],
                          uint2 threadPositionInGrid [[thread_position_in_grid]]) {
    float value = ifftBuffer[threadPositionInGrid.y * outputTexture.get_width() + threadPositionInGrid.x].x;
    
    // clamp to [0, 1] range
    value = saturate(value);
    
    outputTexture.write(float4(float3(value), 1.0f), threadPositionInGrid);
}
