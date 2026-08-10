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

struct Uniforms {
    float2 viewSize;
    float zoom;
    float _pad;
    float2 offset;
};

vertex VertexOut vertexRenderGame(
    uint vertexID [[vertex_id]],
    const device uint8_t* currentGrid [[buffer(0)]],
    constant uint2& gridSizePixels [[buffer(1)]],
    constant Uniforms& uniforms [[buffer(2)]]) {
    VertexOut out;
    // cull dead cells
    if (currentGrid[vertexID] == 0) {
        out.position = float4(-2.0, -2.0, 0.0, 1.0);
        out.pointSize = 0;
        return out;
    }

    uint2 globalPos;
    globalPos.y = vertexID / gridSizePixels.x;
    globalPos.x = vertexID % gridSizePixels.x;

    // screen position
    float2 pixelPos = float2(globalPos);
    float2 adjustedPos = (pixelPos * uniforms.zoom) + uniforms.offset;
    
    // map to clip space (-1.0 to 1.0)
    float x = (adjustedPos.x / uniforms.viewSize.x) * 2.0 - 1.0;
    float y = (1.0 - (adjustedPos.y / uniforms.viewSize.y) * 2.0); // Flip Y

    out.position = float4(x, y, 0.0, 1.0);

    // set pixel size, subtract 1.0 to add a tiny gap between cells
    out.pointSize = max(2.0, uniforms.zoom - 1.0);
    
    out.color = float4(0.0, 1.0, 0.0, 1.0); // Green
    
    return out;
}

// fragment shader
fragment float4 fragmentRenderGame(VertexOut in [[stage_in]]) {
    // return the color calculated in the vertex shader
    return in.color;
}
