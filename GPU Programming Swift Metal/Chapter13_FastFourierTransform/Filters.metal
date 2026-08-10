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

kernel void fftGaussianBlur(device float2* fftData [[buffer(0)]],
                            constant uint& width [[buffer(1)]],
                            constant uint& height [[buffer(2)]],
                            uint2 threadPositionInGrid [[thread_position_in_grid]]) {
    const float sigma = 14.0f;
    const float sigma2 = 2.0f * sigma * sigma;
    
    float fx = (float)min(threadPositionInGrid.x, width - threadPositionInGrid.x);
    float fy = (float)min(threadPositionInGrid.y, height - threadPositionInGrid.y);
    
    float dist2 = fx * fx + fy * fy;
    float mask = fast::exp(-dist2 / sigma2);
    
    uint index = threadPositionInGrid.y * width + threadPositionInGrid.x;
    fftData[index] *= mask;
}
