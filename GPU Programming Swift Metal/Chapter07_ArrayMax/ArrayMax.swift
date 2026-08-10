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

import Metal

private func maxThreadgroupMemory(_ input: inout [Float]) async -> Float {
    let processor = ParallelProcessor()
    
    var inputBuffer = processor.device.makeBuffer(bytesNoCopy: &input, length: MemoryLayout<Float>.stride * input.count, options: .storageModeShared)!
    
    // (1) each threadgroup will reduce over a fixed number of elements on each iteration
    let elementsPerThreadgroup = 32768
    var elementsPerThreadgroupUInt32 = UInt32(elementsPerThreadgroup)
    let elementsPerThreadgroupBuffer = processor.device.makeBuffer(bytesNoCopy: &elementsPerThreadgroupUInt32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    
    let threadsPerThreadgroup = 256
    var count = input.count
    // (2) continue reducing until the remaining elements fit within a single threadgroup's capacity
    while count > elementsPerThreadgroup {
        let countBuffer = processor.device.makeBuffer(bytesNoCopy: &count, length: MemoryLayout<UInt64>.stride, options: .storageModeShared)!
        // (3) calculate the number of threadgroups needed based on current total count and the number of elements per threadgroup
        let numThreadgroups = (count + elementsPerThreadgroup - 1) / elementsPerThreadgroup
        var elementsPerThread = UInt32(elementsPerThreadgroup / threadsPerThreadgroup);
        let elementsPerThreadBuffer = processor.device.makeBuffer(bytesNoCopy: &elementsPerThread, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let outputBuffer = processor.device.makeBuffer(length: MemoryLayout<Float>.stride * numThreadgroups, options: .storageModeShared)!
        
        await processor.process(kernelName: "maxThreadgroupMemory",
                                buffers: inputBuffer, countBuffer, elementsPerThreadgroupBuffer, elementsPerThreadBuffer, outputBuffer,
                                threadgroupMemory: MemoryLayout<Float>.stride * threadsPerThreadgroup,
                                options: ProcessOptions(
                                    threadgroupsPerGrid: numThreadgroups,
                                    threadsPerThreadgroup: threadsPerThreadgroup))
        // (4) the output of this pass is the input of the next pass
        inputBuffer = outputBuffer
        
        // (5) update the count to reflect the number of partial maximums generated for the next reduction pass
        count /= elementsPerThreadgroup
    }
    
    let ptr = inputBuffer.contents().bindMemory(to: Float.self, capacity: count)
    let result = Array(UnsafeBufferPointer(start: ptr, count: count))
    
    // (6) at this point results will have <= elementsPerThreadgroup elements, use the built-in max function to calculate the max on the CPU
    return result.max()!
}

private func maxSingleSimdgroup(_ input: inout [Float]) async -> Float {
    let processor = ParallelProcessor()
    
    var inputBuffer = processor.device.makeBuffer(bytesNoCopy: &input, length: MemoryLayout<Float>.stride * input.count, options: .storageModeShared)!
    
    let elementsPerThreadgroup = 32768
    var elementsPerThreadgroupUInt32 = UInt32(elementsPerThreadgroup)
    let elementsPerThreadgroupBuffer = processor.device.makeBuffer(bytesNoCopy: &elementsPerThreadgroupUInt32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    
    var count = input.count
    let threadsPerThreadgroup = 32
    while count > elementsPerThreadgroup {
        let countBuffer = processor.device.makeBuffer(bytesNoCopy: &count, length: MemoryLayout<UInt64>.stride, options: .storageModeShared)!
        let numThreadgroups = (count + elementsPerThreadgroup - 1) / elementsPerThreadgroup
        let outputBuffer = processor.device.makeBuffer(length: MemoryLayout<Float>.stride * numThreadgroups, options: .storageModeShared)!
        
        await processor.process(kernelName: "maxSingleSimdgroup",
                                buffers: inputBuffer, countBuffer, elementsPerThreadgroupBuffer, outputBuffer,
                                options: ProcessOptions(
                                    threadgroupsPerGrid: numThreadgroups,
                                    threadsPerThreadgroup: threadsPerThreadgroup))
        
        inputBuffer = outputBuffer
        count /= elementsPerThreadgroup
    }
    
    let ptr = inputBuffer.contents().bindMemory(to: Float.self, capacity: count)
    let result = Array(UnsafeBufferPointer(start: ptr, count: count))
    
    return result.max()!
}

func arrayMax() async {
    var dataSizes: [Int] = []
    dataSizes.append(Int(pow(2.0, 15.0)))
    dataSizes.append(Int(pow(2.0, 28.0)))
    dataSizes.append(Int(pow(2.0, 30.0)))
    dataSizes.append(Int(pow(2.0, 32.0)))
    
    for dataSize in dataSizes {
        print("\ngenerating array")
        var startTime = CFAbsoluteTimeGetCurrent()
        var input = await ParallelData.randomFloatArray(length: dataSize)
        print("data size: \(dataSize.formatted()) floats (\(MemoryLayout<Float>.stride * dataSize / 1_000_000) MB)")
        print(String(format: "elapsed time: %.4f seconds\n", CFAbsoluteTimeGetCurrent() - startTime))
        
        //        startTime = CFAbsoluteTimeGetCurrent()
        //        let sequentialMax = input.max()!
        //        print(String(format: "sequentialMax: %f, %.4f seconds\n", sequentialMax, CFAbsoluteTimeGetCurrent() - startTime))
        
        startTime = CFAbsoluteTimeGetCurrent()
        let parallalMaxThreadgroupMemory = await maxThreadgroupMemory(&input)
        print(String(format: "parallalMaxThreadgroupMemory: %f, %.4f seconds", parallalMaxThreadgroupMemory, CFAbsoluteTimeGetCurrent() - startTime))
                
        startTime = CFAbsoluteTimeGetCurrent()
        let parallalMaxSingleSimdgroup = await maxSingleSimdgroup(&input)
        print(String(format: "parallalMaxSingleSimdgroup: %f, %.4f seconds", parallalMaxSingleSimdgroup, CFAbsoluteTimeGetCurrent() - startTime))
    }
}
