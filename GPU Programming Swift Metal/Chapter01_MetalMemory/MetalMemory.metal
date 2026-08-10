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

kernel void memoryPrintInput(const device int* input [[buffer(0)]],
                             uint threadPositionInGrid [[thread_position_in_grid]]) {
    int value = input[threadPositionInGrid];
    os_log_default.log("threadPositionInGrid %u, value %d", threadPositionInGrid, value);
}

kernel void memoryOutputInc(const device int* input [[buffer(0)]],
                            device int* output [[buffer(1)]],
                            uint threadPositionInGrid [[thread_position_in_grid]]) {
    int inputValue = input[threadPositionInGrid];
    int outputValue = inputValue + 1;
    output[threadPositionInGrid] = outputValue;
    os_log_default.log("threadPositionInGrid %u, input value %d, output value: %d", threadPositionInGrid, inputValue, outputValue);
}

kernel void memoryOutputIncThreadgroupMemory(const device int* input [[buffer(0)]],
                            device int* output [[buffer(1)]],
                            threadgroup int* sharedMemory [[threadgroup(0)]],
                            uint threadPositionInGrid [[thread_position_in_grid]]) {
    // `inputValue` and `outputValue` are declared inside the kernel and use thread address space
    int inputValue = input[threadPositionInGrid];
    int outputValue = inputValue + 1;
    sharedMemory[threadPositionInGrid] = outputValue;
    output[threadPositionInGrid] = sharedMemory[threadPositionInGrid];
    os_log_default.log("threadPositionInGrid %u, input value %d, output value: %d", threadPositionInGrid, inputValue, outputValue);
}

constant int incrementValue = 10;

kernel void memoryOutputIncConstant(const device int* input [[buffer(0)]],
                                    device int* output [[buffer(1)]],
                                    uint threadPositionInGrid [[thread_position_in_grid]]) {
    int inputValue = input[threadPositionInGrid];
    int outputValue = inputValue + incrementValue;
    output[threadPositionInGrid] = outputValue;
    os_log_default.log("threadPositionInGrid %u, input value %d, output value: %d", threadPositionInGrid, inputValue, outputValue);
}

constant int functionConstantIncrementValue[[function_constant(0)]];

kernel void memoryOutputIncFunctionConstant(const device int* input [[buffer(0)]],
                                            device int* output [[buffer(1)]],
                                            uint threadPositionInGrid [[thread_position_in_grid]]) {
    int inputValue = input[threadPositionInGrid];
    int outputValue = inputValue + functionConstantIncrementValue;
    output[threadPositionInGrid] = outputValue;
    os_log_default.log("threadPositionInGrid %u, input value %d, output value: %d", threadPositionInGrid, inputValue, outputValue);
}

kernel void memoryHeap(const device int* input [[buffer(0)]],
                       device int* output1 [[buffer(1)]],
                       device int* output2 [[buffer(2)]],
                       uint threadPositionInGrid [[thread_position_in_grid]]) {
    int inputValue = input[threadPositionInGrid];
    int outputValue1 = inputValue + 1;
    output1[threadPositionInGrid] = outputValue1;
    int outputValue2 = inputValue * 2;
    output2[threadPositionInGrid] = outputValue2;
    os_log_default.log("threadPositionInGrid %u, input value %d, output value 1: %d, output value 2: %d", threadPositionInGrid, inputValue, outputValue1, outputValue2);
}

kernel void memoryBandwidth(const device float4 *input [[buffer(0)]],
                            device float4 *output [[buffer(1)]],
                            uint threadPositionInGrid [[thread_position_in_grid]]) {
    output[threadPositionInGrid] = input[threadPositionInGrid];
}

kernel void memoryRandomRead(const device float *input [[buffer(0)]],
                             const device int* indices [[buffer(1)]],
                             constant uint& chunkLength [[buffer(2)]],
                             device float *output [[buffer(3)]],
                             uint threadPositionInGrid [[thread_position_in_grid]]) {
    uint start = chunkLength * threadPositionInGrid;
    uint end = start + chunkLength;
    for(uint i = start; i < end; i++) {
        // (1) read from input buffer based on the given indices, write in contiguous chunks
        output[i] = input[indices[i]];
    }
}

kernel void memoryRandomWrite(const device float *input [[buffer(0)]],
                              const device int* indices [[buffer(1)]],
                              constant uint& chunkLength [[buffer(2)]],
                              device float *output [[buffer(3)]],
                              uint threadPositionInGrid [[thread_position_in_grid]]) {
    uint start = chunkLength * threadPositionInGrid;
    uint end = start + chunkLength;
    for(uint i = start; i < end; i++) {
        // (1) read in contiguous chunks, write to output buffer based on the given indices
        output[indices[i]] = input[i];
    }
}

kernel void memoryRandomReadWrite(const device float *input [[buffer(0)]],
                                  const device int* indices [[buffer(1)]],
                                  constant uint& chunkLength [[buffer(2)]],
                                  device float *output [[buffer(3)]],
                                  uint threadPositionInGrid [[thread_position_in_grid]]) {
    uint start = chunkLength * threadPositionInGrid;
    uint end = start + chunkLength;
    for(uint i = start; i < end; i++) {
        // (1) read from input buffer and write to output buffer based on the given indices
        output[indices[i]] = input[indices[i]];
    }
}
