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

kernel void mandelbrot(const device uchar* redMap [[buffer(0)]],
                       const device uchar* greenMap [[buffer(1)]],
                       const device uchar* blueMap [[buffer(2)]],
                       constant float2& min [[buffer(3)]],
                       constant float2& scale [[buffer(4)]],
                       constant uint& width [[buffer(5)]],
                       constant uint& height [[buffer(6)]],
                       device uchar* pixels [[buffer(7)]],
                       uint2 threadPositionInGrid [[thread_position_in_grid]]) {
    float cReal = min.x + (float(threadPositionInGrid.x) * scale.x);
    float cImaginary = min.y + (float(threadPositionInGrid.y) * scale.y);

    float zReal = 0.0;
    float zImaginary = 0.0;
    int colorMapIndex = 0;
    while (colorMapIndex < 255) {
        // Z_n+1 = Z_n^2 + C
        float realSquared = zReal * zReal;
        float imaginarySquared = zImaginary * zImaginary;
        float zNextReal = cReal + (realSquared - imaginarySquared);
        float zNextImaginary = cImaginary + (2.0 * zReal * zImaginary);
        float lengthSquared = realSquared + imaginarySquared;
        if (lengthSquared >= 4.0) {
            break;
        }
        zReal = zNextReal;
        zImaginary = zNextImaginary;
        colorMapIndex += 1;
    }
    
    int pixelIndex = (threadPositionInGrid.y * width + threadPositionInGrid.x) * 3;
    pixels[pixelIndex] = redMap[colorMapIndex];
    pixels[pixelIndex + 1] = greenMap[colorMapIndex];
    pixels[pixelIndex + 2] = blueMap[colorMapIndex];
}
