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

kernel void ifThenScalar(device uint* buffer [[buffer(0)]],
                          uint threadPositionInGrid [[thread_position_in_grid]]) {
    bool condition = threadPositionInGrid % 2 == 1;
    uint falseValue = threadPositionInGrid;
    uint trueValue = threadPositionInGrid * 2;
    uint x;
    if (condition) {
        x = trueValue;
    } else {
        x = falseValue;
    }
    buffer[threadPositionInGrid] = x;
}

float heavyCalculation(float val) {
    float result = val;
    for(int i = 0; i < 10; ++i) {
        result = sin(result) * 2.0f;
    }
    return result;
}

kernel void ternaryScalar(device float* buffer [[buffer(0)]],
                          uint threadPositionInGrid [[thread_position_in_grid]]) {
    float input = buffer[threadPositionInGrid];
    bool condition = input > 0.5f;
    float output = condition ? heavyCalculation(input) : 0.0f;
    buffer[threadPositionInGrid] = output;
}

kernel void selectScalar(device float* buffer [[buffer(0)]],
                         uint threadPositionInGrid [[thread_position_in_grid]]) {
    float input = buffer[threadPositionInGrid];
    bool condition = input > 0.5f;
    float output = select(0.0f, heavyCalculation(input), condition);
    buffer[threadPositionInGrid] = output;
}

float4 heavyVectorCalculation(float4 val) {
    float4 result = val;
    for(int i = 0; i < 10; ++i) {
        result = sin(result) * 2.0f;
    }
    return result;
}

kernel void selectVector(device float4* buffer [[buffer(0)]],
                         uint threadPositionInGrid [[thread_position_in_grid]]) {
    float4 input = buffer[threadPositionInGrid];
    bool4 condition = input > 0.5f;
    float4 output = select(0.0f, heavyVectorCalculation(input), condition);
    buffer[threadPositionInGrid] = output;
}
