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

struct VertexOut {
    float4 position [[position]];
    float pointSize [[point_size]];
    float4 color;
};

float3 getEnvironmentColor(uint id) {
    float3 colors[8] = {
        float3(1.0, 0.0, 0.0),
        float3(1.0, 0.5, 0.0),
        float3(1.0, 1.0, 0.0),
        float3(0.0, 1.0, 0.0),
        float3(0.0, 1.0, 1.0),
        float3(0.0, 0.5, 1.0),
        float3(0.0, 0.0, 1.0),
        float3(1.0, 0.0, 1.0)
    };
    return colors[id % 8];
}

vertex VertexOut vertexRenderGrayScott(uint vertexID [[vertex_id]],
                                       const device float* currentV [[buffer(0)]],
                                       constant uint2& gridSize [[buffer(1)]],
                                       constant float2& viewSize [[buffer(2)]],
                                       constant uint& totalEnvironments [[buffer(3)]],
                                       constant uint* cellEnvironments [[buffer(4)]]) {
    VertexOut out;
    
    // map vertex ID to cell grid coordinate
    uint xGrid = vertexID % gridSize.x;
    uint yGrid = vertexID / gridSize.x;
    uint environmentID = cellEnvironments[vertexID];
    float threadLocalV = currentV[vertexID];
    
    // convert grid space to clip space, map 0..width to -1..1
    float xNorm = (float(xGrid) / float(gridSize.x)) * 2.0 - 1.0;
    float yNorm = (1.0 - (float(yGrid) / float(gridSize.y))) * 2.0 - 1.0; // flip Y
    
    out.position = float4(xNorm, yNorm, 0.0, 1.0);
    
    float scaleX = viewSize.x / float(gridSize.x);
    float scaleY = viewSize.y / float(gridSize.y);
    out.pointSize = ceil(max(scaleX, scaleY));

    float3 environmentsColor = getEnvironmentColor(environmentID);
    out.color = float4(environmentsColor * threadLocalV * 3.5, 1.0);
    
    return out;
}

fragment float4 fragmentRenderGrayScott(VertexOut in [[stage_in]]) {
    return in.color;
}
