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

nonisolated private func sequentialFor(input: [Float]) -> Float {
    var sum: Double = 0.0
    for value in input {
        sum += Double(value)
    }
    return Float(sum)
}

private func sumDeviceMemory(input: inout [Float], threadsPerThreadgroup: Int) async -> Float {
    let processor = ParallelProcessor()

    let kernelName = "sumDeviceMemory"
    let computePipelineState = await processor.registerKernel(kernelName)
    let threadCount = min(threadsPerThreadgroup, computePipelineState.maxTotalThreadsPerThreadgroup)

    // (1) create a shared buffer that points to the contents of the input array
    let inputBuffer = processor.device.makeBuffer(bytesNoCopy: &input, length: MemoryLayout<Float>.stride * input.count, options: .storageModeShared)!

    // (2) create a buffer for the length of the input array
    var countU64 = UInt64(input.count)
    let countBuffer = processor.device.makeBuffer(bytes: &countU64, length: MemoryLayout<UInt64>.stride, options: .storageModeShared)!

    // (3) create an output buffer with enough capacity to hold the partial sums for each thread
    let outputBuffer = processor.device.makeBuffer(length: MemoryLayout<Float>.stride * threadCount, options: .storageModeShared)!
    
    // (4) threadsPerThreadgroup and threadsPerGrid are equal
    let options = ProcessOptions(threadsPerGrid: threadCount, threadsPerThreadgroup: threadCount)
    
    // (5) launch the kernel
    await processor.process(kernelName: kernelName,
                            buffers: inputBuffer, countBuffer, outputBuffer,
                            options: options)
    
    // (6) bind the first `Float.size` bytes in the output buffer to a Swift Float
    let result = outputBuffer.contents().bindMemory(to: Float.self, capacity: 1).pointee
    return result
}

private func sumThreadgroupMemory(input: inout [Float], threadsPerThreadgroup: Int) async -> Float {
    let processor = ParallelProcessor()
    
    let kernelName = "sumThreadgroupMemory"
    let computePipelineState = await processor.registerKernel(kernelName)
    let threadCount = min(threadsPerThreadgroup, computePipelineState.maxTotalThreadsPerThreadgroup)

    let inputBuffer = processor.device.makeBuffer(bytesNoCopy: &input, length: MemoryLayout<Float>.stride * input.count, options: .storageModeShared)!
    
    // (1) output buffer only holds a single float, partial sums use threadgroup memory
    var countU64 = UInt64(input.count)
    let countBuffer = processor.device.makeBuffer(bytes: &countU64, length: MemoryLayout<UInt64>.stride, options: .storageModeShared)!

    let outputBuffer = processor.device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)!

    let options = ProcessOptions(threadsPerGrid: threadCount, threadsPerThreadgroup: threadCount)
    await processor.process(kernelName: "sumThreadgroupMemory",
                            buffers: inputBuffer, countBuffer, outputBuffer,
                            // (2) define threadgroup memory
                            threadgroupMemory: MemoryLayout<Float>.stride * threadCount,
                            options: options)
    
    let result = outputBuffer.contents().bindMemory(to: Float.self, capacity: 1).pointee
    return result
}

private func sumThreadgroupMemoryGridStride(input: inout [Float], threadsPerThreadgroup: Int) async -> Float {
    let processor = ParallelProcessor()
    
    let kernelName = "sumThreadgroupMemoryGridStride"
    let computePipelineState = await processor.registerKernel(kernelName)
    let threadCount = min(threadsPerThreadgroup, computePipelineState.maxTotalThreadsPerThreadgroup)

    let inputBuffer = processor.device.makeBuffer(bytesNoCopy: &input, length: MemoryLayout<Float>.stride * input.count, options: .storageModeShared)!

    var countU64 = UInt64(input.count)
    let countBuffer = processor.device.makeBuffer(bytes: &countU64, length: MemoryLayout<UInt64>.stride, options: .storageModeShared)!

    let outputBuffer = processor.device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)!

    let options = ProcessOptions(threadsPerGrid: threadCount, threadsPerThreadgroup: threadCount)
    await processor.process(kernelName: kernelName,
                            buffers: inputBuffer, countBuffer, outputBuffer,
                            threadgroupMemory: MemoryLayout<Float>.stride * threadCount,
                            options: options)
    
    let result = outputBuffer.contents().bindMemory(to: Float.self, capacity: 1).pointee
    return result
}

private func sumAtomics(input: inout [Float], threadsPerThreadgroup: Int) async -> Float {
    let processor = ParallelProcessor()
    
    let kernelName = "sumAtomics"
    let computePipelineState = await processor.registerKernel(kernelName)
    let threadCount = min(threadsPerThreadgroup, computePipelineState.maxTotalThreadsPerThreadgroup)
    
    let inputBuffer = processor.device.makeBuffer(bytesNoCopy: &input, length: MemoryLayout<Float>.stride * input.count, options: .storageModeShared)!

    var countU64 = UInt64(input.count)
    let countBuffer = processor.device.makeBuffer(bytes: &countU64, length: MemoryLayout<UInt64>.stride, options: .storageModeShared)!

    let outputBuffer = processor.device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)!

    let options = ProcessOptions(threadsPerGrid: threadCount, threadsPerThreadgroup: threadCount)
    await processor.process(kernelName: kernelName,
                            buffers: inputBuffer, countBuffer, outputBuffer,
                            options: options)
    
    let result = outputBuffer.contents().bindMemory(to: Float.self, capacity: 1).pointee
    return result
}

private func sumDeviceMemoryMultiThreadgroup(input: inout [Float],
                                             partitionLength: Int,
                                             threadsPerThreadgroupIn: Int) async -> Float {
    let device = MTLCreateSystemDefaultDevice()!
    
    let library = try! device.makeDefaultLibrary(bundle: .main)
    let kernelName = "sumDeviceMemoryMultiThreadgroup"
    let kernelFunction = library.makeFunction(name: kernelName)!
    let computePipelineState = try! await device.makeComputePipelineState(function: kernelFunction)
    let threadsPerThreadgroup = min(threadsPerThreadgroupIn, computePipelineState.maxTotalThreadsPerThreadgroup)

    let inputBuffer = device.makeBuffer(bytesNoCopy: &input, length: MemoryLayout<Float>.stride * input.count, options: .storageModeShared)!

    var partitionLengthU64 = UInt64(partitionLength)
    let partitionLengthBuffer = device.makeBuffer(bytes: &partitionLengthU64, length: MemoryLayout<UInt64>.stride, options: .storageModeShared)!
    
    var count = input.count
    
    // (1) determine total thread count based on input array count and partition length
    var threadsPerGrid = (count + partitionLength - 1) / partitionLength
    
    // (2) create two output buffers
    let outputBuffer1 = device.makeBuffer(length: MemoryLayout<Float>.stride * threadsPerGrid, options: .storageModeShared)!
    let outputBuffer2 = device.makeBuffer(length: MemoryLayout<Float>.stride * threadsPerGrid, options: .storageModeShared)!

    let argumentTableDescriptor = MTL4ArgumentTableDescriptor()
    argumentTableDescriptor.maxBufferBindCount = 4
    
    let residencySet = try! device.makeResidencySet(descriptor: .init())
    residencySet.addAllocation(inputBuffer)
    residencySet.addAllocation(partitionLengthBuffer)
    residencySet.addAllocation(outputBuffer1)
    residencySet.addAllocation(outputBuffer2)
    residencySet.commit()
    
    let commandBuffer = device.makeCommandBuffer()!
    
    let commandAllocator = device.makeCommandAllocator()!
    commandBuffer.beginCommandBuffer(allocator: commandAllocator, options: .init())
    
    var i = 0
    while true {
        // (1) allocate the input count buffer for each iteration
        var countU64 = UInt64(count)
        let countBuffer = device.makeBuffer(bytes: &countU64, length: MemoryLayout<UInt64>.stride, options: .storageModeShared)!
        residencySet.addAllocation(countBuffer)
        residencySet.commit()
        
        let argumentTable = try! device.makeArgumentTable(descriptor: argumentTableDescriptor)
        argumentTable.setAddress(countBuffer.gpuAddress, index: 1)
        argumentTable.setAddress(partitionLengthBuffer.gpuAddress, index: 2)
        if i == 0 {
            // (2) pass the entire input array for iteration 0
            argumentTable.setAddress(inputBuffer.gpuAddress, index: 0)
            argumentTable.setAddress(outputBuffer1.gpuAddress, index: 3)
        } else if i % 2 == 1 {
            // (3) use outputBuffer1 as input and outputBuffer2 as output for iterations 1, 3, 5...
            argumentTable.setAddress(outputBuffer1.gpuAddress, index: 0)
            argumentTable.setAddress(outputBuffer2.gpuAddress, index: 3)
        } else {
            // (4) use outputBuffer2 as input and outputBuffer1 as output for iterations 2, 4, 6...
            argumentTable.setAddress(outputBuffer2.gpuAddress, index: 0)
            argumentTable.setAddress(outputBuffer1.gpuAddress, index: 3)
        }
        
        let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
        computeCommandEncoder.setComputePipelineState(computePipelineState)
        computeCommandEncoder.setArgumentTable(argumentTable)
        // (5) use a producer barrier to synchronize read/writes to output buffers between iterations
        computeCommandEncoder.barrier(afterStages: .dispatch, beforeQueueStages: .dispatch)
        computeCommandEncoder.dispatchThreads(threadsPerGrid: MTLSizeMake(threadsPerGrid, 1, 1),
                                              threadsPerThreadgroup: MTLSizeMake(threadsPerThreadgroup, 1, 1))
        computeCommandEncoder.endEncoding()
        
        i += 1
        // (6) update the input count for next iteration to the total thread count for this iteration
        count = threadsPerGrid
        // (7) divide total thread count by partition length for next iteration
        threadsPerGrid = (threadsPerGrid + partitionLength - 1) / partitionLength
        
        if threadsPerGrid < threadsPerThreadgroup {
            // (8) terminate the loop if the calculated thread count is less than the specified threads per threadgroup
            break
        }
    }
    
    commandBuffer.endCommandBuffer()

    let commandQueue = device.makeMTL4CommandQueue()!
    commandQueue.addResidencySet(residencySet)
    let commitOptions = MTL4CommitOptions()
    
    try! await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
        commitOptions.addFeedbackHandler { feedback in
            if let error = feedback.error {
                continuation.resume(throwing: error)
            } else {
                continuation.resume()
            }
        }
        commandQueue.commit([commandBuffer], options: commitOptions)
    }
    
    var resultPtr: UnsafeMutablePointer<Float>
    if i % 2 == 0 {
        // (1) if an even number of loop iterations were run then the latest partial sums are in outputBuffer2
        resultPtr = outputBuffer2.contents().bindMemory(to: Float.self, capacity: count)
    } else {
        // (2) if an odd number of loop iterations were run then the latest partial sums are in outputBuffer1
        resultPtr = outputBuffer1.contents().bindMemory(to: Float.self, capacity: count)
    }

    // (3) call the single threadgroup sumDeviceMemory kernel to sum the final partial sums
    var partialSums = Array(UnsafeBufferPointer(start: resultPtr, count: count))
    return await sumDeviceMemory(input: &partialSums, threadsPerThreadgroup: threadsPerThreadgroupIn)
}

private func sumAtomicsMultiThreadgroup(input: inout [Float],
                                        threadgroups: Int,
                                        threadsPerThreadgroupIn: Int) async -> Float {
    let processor = ParallelProcessor()
    
    let kernelName = "sumAtomicsMultiThreadgroup"
    let computePipelineState = await processor.registerKernel(kernelName)
    let threadsPerThreadgroup = min(threadsPerThreadgroupIn, computePipelineState.maxTotalThreadsPerThreadgroup)
    
    let inputBuffer = processor.device.makeBuffer(bytesNoCopy: &input, length: MemoryLayout<Float>.stride * input.count, options: .storageModeShared)!
    
    var countU64 = UInt64(input.count)
    let countBuffer = processor.device.makeBuffer(bytes: &countU64, length: MemoryLayout<UInt64>.stride, options: .storageModeShared)!
    
    let outputBuffer = processor.device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)!
    
    // (1) threads per grid is based on the number of threadgroups and threads per threadgroup
    let threadsPerGrid = threadgroups * threadsPerThreadgroup
    let options = ProcessOptions(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    await processor.process(kernelName: kernelName,
                            buffers: inputBuffer, countBuffer, outputBuffer,
                            options: options)
    
    let result = outputBuffer.contents().bindMemory(to: Float.self, capacity: 1).pointee
    return result
}

private func sumAtomicsSIMDReduction(input: inout [Float],
                                     threadgroups: Int,
                                     threadsPerThreadgroupIn: Int) async -> Float {
    let processor = ParallelProcessor()

    let kernelName = "sumAtomicsSIMDReduction"
    let computePipelineState = await processor.registerKernel(kernelName)
    let threadsPerThreadgroup = min(threadsPerThreadgroupIn, computePipelineState.maxTotalThreadsPerThreadgroup)

    let inputBuffer = processor.device.makeBuffer(bytesNoCopy: &input, length: MemoryLayout<Float>.stride * input.count, options: .storageModeShared)!

    var countU64 = UInt64(input.count)
    let countBuffer = processor.device.makeBuffer(bytes: &countU64, length: MemoryLayout<UInt64>.stride, options: .storageModeShared)!

    let outputBuffer = processor.device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)!
    
    let threadsPerGrid = threadgroups * threadsPerThreadgroup
    let options = ProcessOptions(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    await processor.process(kernelName: kernelName,
                            buffers: inputBuffer, countBuffer, outputBuffer,
                            options: options)
    
    let result = outputBuffer.contents().bindMemory(to: Float.self, capacity: 1).pointee
    return result
}

private func sumAtomicsSIMDReductionUnrolled(input: inout [Float],
                                             threadgroups: Int,
                                             threadsPerThreadgroupIn: Int) async -> Float {
    let processor = ParallelProcessor()

    let kernelName = "sumAtomicsSIMDReductionUnrolled"
    let computePipelineState = await processor.registerKernel(kernelName)
    let threadsPerThreadgroup = min(threadsPerThreadgroupIn, computePipelineState.maxTotalThreadsPerThreadgroup)

    let inputBuffer = processor.device.makeBuffer(bytesNoCopy: &input, length: MemoryLayout<Float>.stride * input.count, options: .storageModeShared)!

    var countU64 = UInt64(input.count)
    let countBuffer = processor.device.makeBuffer(bytes: &countU64, length: MemoryLayout<UInt64>.stride, options: .storageModeShared)!

    let outputBuffer = processor.device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)!
    
    let threadsPerGrid = threadgroups * threadsPerThreadgroup
    let options = ProcessOptions(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    await processor.process(kernelName: kernelName,
                            buffers: inputBuffer, countBuffer, outputBuffer,
                            options: options)
    
    let result = outputBuffer.contents().bindMemory(to: Float.self, capacity: 1).pointee
    return result
}

private func sumAtomicsSIMDSum(input: inout [Float],
                               threadgroups: Int,
                               threadsPerThreadgroupIn: Int) async -> Float {
    let processor = ParallelProcessor()
    
    let kernelName = "sumAtomicsSIMDSum"
    let computePipelineState = await processor.registerKernel(kernelName)
    let threadsPerThreadgroup = min(threadsPerThreadgroupIn, computePipelineState.maxTotalThreadsPerThreadgroup)

    let inputBuffer = processor.device.makeBuffer(bytesNoCopy: &input, length: MemoryLayout<Float>.stride * input.count, options: .storageModeShared)!

    var countU64 = UInt64(input.count)
    let countBuffer = processor.device.makeBuffer(bytes: &countU64, length: MemoryLayout<UInt64>.stride, options: .storageModeShared)!

    let outputBuffer = processor.device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)!
    
    let threadsPerGrid = threadgroups * threadsPerThreadgroup
    let options = ProcessOptions(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    await processor.process(kernelName: kernelName,
                            buffers: inputBuffer, countBuffer, outputBuffer,
                            options: options)
    
    let result = outputBuffer.contents().bindMemory(to: Float.self, capacity: 1).pointee
    return result
}

private func sumAtomicsSIMDSumPackedFloat(input: inout [Float],
                                          threadgroups: Int,
                                          threadsPerThreadgroupIn: Int) async -> Float {
    let processor = ParallelProcessor()

    let kernelName = "sumAtomicsSIMDSumPackedFloat"
    let computePipelineState = await processor.registerKernel(kernelName)
    let threadsPerThreadgroup = min(threadsPerThreadgroupIn, computePipelineState.maxTotalThreadsPerThreadgroup)

    // (1) instantiante the input buffer as SIMD4<Float>, note the division by 4
    let inputBuffer = processor.device.makeBuffer(bytesNoCopy: &input, length: MemoryLayout<SIMD4<Float>>.stride * input.count / 4, options: .storageModeShared)!

    // (2) divide the input by 4 since the values are now packed
    var countU64 = UInt64(input.count / 4)
    let countBuffer = processor.device.makeBuffer(bytes: &countU64, length: MemoryLayout<UInt64>.stride, options: .storageModeShared)!

    let outputBuffer = processor.device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)!
    
    let threadsPerGrid = threadgroups * threadsPerThreadgroup
    let options = ProcessOptions(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    await processor.process(kernelName: kernelName,
                            buffers: inputBuffer, countBuffer, outputBuffer,
                            options: options)
    
    let result = outputBuffer.contents().bindMemory(to: Float.self, capacity: 1).pointee
    return result
}

nonisolated func arraySum() async {
    var dataSizes: [Int] = []
    dataSizes.append(Int(pow(2.0, 28.0)))
    dataSizes.append(Int(pow(2.0, 30.0)))
    dataSizes.append(Int(pow(2.0, 32.0)))
    
    for dataSize in dataSizes {
        print("\ngenerating array")
        var startTime = CFAbsoluteTimeGetCurrent()
        var input = await ParallelData.randomFloatArray(length: dataSize)
        print("data size: \(dataSize.formatted()) floats (\(MemoryLayout<Float>.stride * dataSize / 1_000_000) MB)")
        print(String(format: "elapsed time: %.4f seconds\n", CFAbsoluteTimeGetCurrent() - startTime))
        
        let threadsPerThreadgroup = [32, 64, 128, 256, 512, 1024]

        let threadgroups = (6...16).map { 1 << $0 }
        let partitionLength = (6...18).map { 1 << $0 }

//        startTime = CFAbsoluteTimeGetCurrent()
//        let sum = sequentialFor(input: input)
//        print(String(format: "sequentialFor: %f, %.4f seconds\n", sum, CFAbsoluteTimeGetCurrent() - startTime))
        
        for tpt in threadsPerThreadgroup {
            startTime = CFAbsoluteTimeGetCurrent()
            var sum = await sumDeviceMemory(input: &input, threadsPerThreadgroup: tpt)
            print(String(format: "sumDeviceMemory threadsPerThreadgroup \(tpt): %f, %.4f seconds\n", sum, CFAbsoluteTimeGetCurrent() - startTime))
            
            startTime = CFAbsoluteTimeGetCurrent()
            sum = await sumThreadgroupMemory(input: &input, threadsPerThreadgroup: tpt)
            print(String(format: "sumThreadgroupMemory threadsPerThreadgroup \(tpt): %f, %.4f seconds\n", sum, CFAbsoluteTimeGetCurrent() - startTime))
            
            startTime = CFAbsoluteTimeGetCurrent()
            sum = await sumThreadgroupMemoryGridStride(input: &input, threadsPerThreadgroup: tpt)
            print(String(format: "sumThreadgroupMemoryGridStride threadsPerThreadgroup \(tpt): %f, %.4f seconds\n", sum, CFAbsoluteTimeGetCurrent() - startTime))
            
            startTime = CFAbsoluteTimeGetCurrent()
            sum = await sumAtomics(input: &input, threadsPerThreadgroup: tpt)
            print(String(format: "sumAtomics threadsPerThreadgroup \(tpt): %f, %.4f seconds\n", sum, CFAbsoluteTimeGetCurrent() - startTime))
            
            for ept in partitionLength {
                startTime = CFAbsoluteTimeGetCurrent()
                sum = await sumDeviceMemoryMultiThreadgroup(input: &input, partitionLength: ept, threadsPerThreadgroupIn: tpt)
                print(String(format: "sumDeviceMemoryMultiThreadgroup partitionLength \(ept), threadsPerThreadgroup \(tpt): %f, %.4f seconds", sum, CFAbsoluteTimeGetCurrent() - startTime))
            }
            
            for t in threadgroups {
                startTime = CFAbsoluteTimeGetCurrent()
                var sum = await sumAtomicsMultiThreadgroup(input: &input, threadgroups: t, threadsPerThreadgroupIn: tpt)
                print(String(format: "sumAtomicsMultiThreadgroup threadgroups: \(t), threadsPerThreadgroup \(tpt): %f, %.4f seconds", sum, CFAbsoluteTimeGetCurrent() - startTime))

                startTime = CFAbsoluteTimeGetCurrent()
                sum = await sumAtomicsSIMDReduction(input: &input, threadgroups: t, threadsPerThreadgroupIn: tpt)
                print(String(format: "sumAtomicsSIMDReduction threadgroups: \(t), threadsPerThreadgroup \(tpt): %f, %.4f seconds", sum, CFAbsoluteTimeGetCurrent() - startTime))

                startTime = CFAbsoluteTimeGetCurrent()
                sum = await sumAtomicsSIMDReductionUnrolled(input: &input, threadgroups: t, threadsPerThreadgroupIn: tpt)
                print(String(format: "sumAtomicsSIMDReductionUnrolled threadgroups: \(t), threadsPerThreadgroup \(tpt): %f, %.4f seconds", sum, CFAbsoluteTimeGetCurrent() - startTime))

                startTime = CFAbsoluteTimeGetCurrent()
                sum = await sumAtomicsSIMDSum(input: &input, threadgroups: t, threadsPerThreadgroupIn: tpt)
                print(String(format: "sumAtomicsSIMDSum threadgroups: \(t), threadsPerThreadgroup \(tpt): %f, %.4f seconds", sum, CFAbsoluteTimeGetCurrent() - startTime))
                
                startTime = CFAbsoluteTimeGetCurrent()
                sum = await sumAtomicsSIMDSumPackedFloat(input: &input, threadgroups: t, threadsPerThreadgroupIn: tpt)
                print(String(format: "sumAtomicsSIMDSumPackedFloat threadgroups: \(t), threadsPerThreadgroup \(tpt): %f, %.4f seconds", sum, CFAbsoluteTimeGetCurrent() - startTime))
            }
        }
        
        input.removeAll(keepingCapacity: false)
    }
}
