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

kernel void rgbToHsl(device float* H [[buffer(0)]],
                     device float* S [[buffer(1)]],
                     device float2* L [[buffer(2)]],
                     texture2d<float, access::read> inputTexture [[texture(0)]],
                     uint2 threadPositionInGrid [[thread_position_in_grid]]) {
    float4 pixelColor = inputTexture.read(threadPositionInGrid);
    float r = pixelColor.r;
    float g = pixelColor.g;
    float b = pixelColor.b;
    
    float maxChannel = max(r, max(g, b));
    float minChannel = min(r, min(g, b));
    float delta = maxChannel - minChannel;
    
    float hue = 0.0f;
    float saturation = 0.0f;
    float lightness = (maxChannel + minChannel) * 0.5f;
    if (delta > 0.0f) {
        if (lightness > 0.5f) {
            saturation = delta / (2.0f - maxChannel - minChannel);
        } else {
            saturation = delta / (maxChannel + minChannel);
        }
        
        if (maxChannel == r) {
            hue = (g - b) / delta + (g < b ? 6.0f : 0.0f);
        } else if (maxChannel == g) {
            hue = (b - r) / delta + 2.0f;
        } else {
            hue = (r - g) / delta + 4.0f;
        }
        hue /= 6.0f;
    }
    
    uint index = threadPositionInGrid.y * inputTexture.get_width() + threadPositionInGrid.x;
    H[index] = hue;
    S[index] = saturation;
    L[index] = float2(lightness, 0.0f);
}

inline float hueToRgb(float p, float q, float t) {
    if (t < 0.0f) t += 1.0f;
    if (t > 1.0f) t -= 1.0f;
    if (t < 1.0f/6.0f) return p + (q - p) * 6.0f * t;
    if (t < 1.0f/2.0f) return q;
    if (t < 2.0f/3.0f) return p + (q - p) * (2.0f/3.0f - t) * 6.0f;
    return p;
}

kernel void hslToRgb(const device float* H [[buffer(0)]],
                     const device float* S [[buffer(1)]],
                     const device float2* L [[buffer(2)]],
                     texture2d<float, access::write> outputTexture [[texture(0)]],
                     uint2 threadPositionInGrid [[thread_position_in_grid]]) {
    uint index = threadPositionInGrid.y * outputTexture.get_width() + threadPositionInGrid.x;
    float h = H[index];
    float s = S[index];
    // extract real component from ifft and clamp to [0,1]
    float l = clamp(L[index].x, 0.0f, 1.0f);
    
    float3 rgb;
    if (s == 0.0f) {
        rgb = float3(l);
    } else {
        float q;
        if(l < 0.5f) {
            q = l * (1.0f + s);
        } else {
            q = l + s - l * s;
        }
        float p = 2.0f * l - q;
        
        rgb.r = hueToRgb(p, q, h + 1.0f/3.0f);
        rgb.g = hueToRgb(p, q, h);
        rgb.b = hueToRgb(p, q, h - 1.0f/3.0f);
    }
    
    outputTexture.write(float4(rgb, 1.0f), threadPositionInGrid);
}

kernel void rgbToYCrCb(device float2* Y [[buffer(0)]],
                       device float* Cr [[buffer(1)]],
                       device float* Cb [[buffer(2)]],
                       texture2d<float, access::read> inputTexture [[texture(0)]],
                       uint2 threadPositionInGrid [[thread_position_in_grid]]) {
    float4 rgb = inputTexture.read(threadPositionInGrid);
    
    uint index = threadPositionInGrid.y * inputTexture.get_width() + threadPositionInGrid.x;
    // BT.601 conversion
    Y[index] = float2(0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b, 0.0);
    Cr[index] = 0.5 * rgb.r - 0.418688 * rgb.g - 0.081312 * rgb.b + 0.5;
    Cb[index] =  -0.168736 * rgb.r - 0.331264 * rgb.g + 0.5 * rgb.b + 0.5;
}

kernel void yCrCbToRgb(const device float2* Y [[buffer(0)]],
                       const device float* Cr [[buffer(1)]],
                       const device float* Cb [[buffer(2)]],
                       texture2d<float, access::write> outputTexture [[texture(0)]],
                       uint2 threadPositionInGrid [[thread_position_in_grid]]) {
    uint index = threadPositionInGrid.y * outputTexture.get_width() + threadPositionInGrid.x;
    
    // extract real component from ifft and clamp to [0,1]
    float y = clamp(Y[index].x, 0.0f, 1.0f);
    
    // shift chrominance to signed range
    float shiftedCr = Cr[index] - 0.5f;
    float shiftedCb = Cb[index] - 0.5f;
    
    float r = y + 1.402f * shiftedCr;
    float g = y - 0.344136f * shiftedCb - 0.714136f * shiftedCr;
    float b = y + 1.772f * shiftedCb;
    
    outputTexture.write(float4(saturate(r), saturate(g), saturate(b), 1.0f), threadPositionInGrid);
}
