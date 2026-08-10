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

private func checkSorted(_ data: [Int32]) {
    var a = data[0]
    for i in 1..<data.count {
        if data[i] < a {
            fatalError("data not sorted \(i) \(data[i]) < \(a)")
        }
        a = data[i]
    }
}

private func mergeSortOddEven(data: inout [Int32]) async -> [Int32] {
    let device = MTLCreateSystemDefaultDevice()!
    let library = try! device.makeDefaultLibrary(bundle: .main)
    
    let dataBuffer = device.makeBuffer(bytesNoCopy: &data, length: MemoryLayout<Int32>.stride * data.count, options: .storageModeShared)!
    
    let residencySet = try! device.makeResidencySet(descriptor: .init())
    residencySet.addAllocation(dataBuffer)
    residencySet.commit()
    
    let commandAllocator = device.makeCommandAllocator()!
    let commandBuffer = device.makeCommandBuffer()!
    commandBuffer.beginCommandBuffer(allocator: commandAllocator, options: .init())
    
    let kernelFunction = library.makeFunction(name: "mergeSortOddEven")!
    let computePipelineState = try! await device.makeComputePipelineState(function: kernelFunction)
    let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoder.setComputePipelineState(computePipelineState)
    let argumentTableDescriptor = MTL4ArgumentTableDescriptor()
    argumentTableDescriptor.maxBufferBindCount = 3
    
    var mergeSize: UInt32 = 2
    // (1) outer merge size loop, start at 2, double until N
    while mergeSize <= data.count {
        var stride = mergeSize / 2
        // (2) inner merge loop, stride starts at mergeSize / 2, halves until 1
        while stride > 0 {
            // (3) set the merge size and stride for this iteration
            let mergeSizeBuffer = device.makeBuffer(bytes: &mergeSize, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
            let strideBuffer = device.makeBuffer(bytes: &stride, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
            residencySet.addAllocation(mergeSizeBuffer)
            residencySet.addAllocation(strideBuffer)
            residencySet.commit()
            
            let argumentTable = try! device.makeArgumentTable(descriptor: argumentTableDescriptor)
            argumentTable.setAddress(dataBuffer.gpuAddress, index: 0)
            argumentTable.setAddress(mergeSizeBuffer.gpuAddress, index: 1)
            argumentTable.setAddress(strideBuffer.gpuAddress, index: 2)
            
            computeCommandEncoder.setArgumentTable(argumentTable)
            // (4) use an intra-pass barrier to prevent next dispatch from running until the current one completes
            computeCommandEncoder.barrier(afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch)
            
            // (5) dispatch N / 2 threads
            computeCommandEncoder.dispatchThreads(
                threadsPerGrid: MTLSizeMake(data.count/2, 1, 1),
                threadsPerThreadgroup: MTLSizeMake(computePipelineState.maxTotalThreadsPerThreadgroup, 1, 1))
            
            stride /= 2
        }
        mergeSize *= 2
    }
    
    computeCommandEncoder.endEncoding()
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
    
    return data
}

private func mergeSortOddEvenThreadgroup(data: inout [Int32]) async -> [Int32] {
    let device = MTLCreateSystemDefaultDevice()!
    let library = try! device.makeDefaultLibrary(bundle: .main)
    
    let dataBuffer = device.makeBuffer(bytesNoCopy: &data, length: MemoryLayout<Int32>.stride * data.count, options: .storageModeShared)!
    
    let residencySet = try! device.makeResidencySet(descriptor: .init())
    residencySet.addAllocation(dataBuffer)
    residencySet.commit()
    
    let commandAllocator = device.makeCommandAllocator()!
    let commandBuffer = device.makeCommandBuffer()!
    commandBuffer.beginCommandBuffer(allocator: commandAllocator, options: .init())
    
    let kernelFunctionThreadgroup = library.makeFunction(name: "mergeSortOddEvenThreadgroup")!
    let computePipelineStateThreadgroup = try! await device.makeComputePipelineState(function: kernelFunctionThreadgroup)
    let computeCommandEncoderThreadgroup = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoderThreadgroup.setComputePipelineState(computePipelineStateThreadgroup)
    
    let argumentTableDescriptorThreadgroup = MTL4ArgumentTableDescriptor()
    argumentTableDescriptorThreadgroup.maxBufferBindCount = 1
    let argumentTableThreadgroup = try! device.makeArgumentTable(descriptor: argumentTableDescriptorThreadgroup)
    argumentTableThreadgroup.setAddress(dataBuffer.gpuAddress, index: 0)
    computeCommandEncoderThreadgroup.setArgumentTable(argumentTableThreadgroup)
    
    // (1) dispatch the threadgroup memory kernel with N / 2 threads
    computeCommandEncoderThreadgroup.dispatchThreads(
        threadsPerGrid: MTLSizeMake(data.count/2, 1, 1),
        threadsPerThreadgroup: MTLSizeMake(512, 1, 1))
    // (2) use a producer barrier to prevent the device memory kernel dispatches from running until this one completes
    computeCommandEncoderThreadgroup.barrier(afterStages: .dispatch,
                                             beforeQueueStages: .dispatch,
                                             visibilityOptions: .device)
    computeCommandEncoderThreadgroup.endEncoding()
    
    let kernelFunction = library.makeFunction(name: "mergeSortOddEven")!
    let computePipelineState = try! await device.makeComputePipelineState(function: kernelFunction)
    let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoder.setComputePipelineState(computePipelineState)
    let argumentTableDescriptor = MTL4ArgumentTableDescriptor()
    argumentTableDescriptor.maxBufferBindCount = 3
    
    // (3) the merge size now starts at 2048, the 2 to 1024 merge sizes are handled by the threadgroup memory kernel
    var mergeSize: UInt32 = 2048
    while mergeSize <= data.count {
        var stride = mergeSize / 2
        while stride > 0 {
            let mergeSizeBuffer = device.makeBuffer(bytes: &mergeSize, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
            let strideBuffer = device.makeBuffer(bytes: &stride, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
            residencySet.addAllocation(mergeSizeBuffer)
            residencySet.addAllocation(strideBuffer)
            residencySet.commit()
            
            let argumentTable = try! device.makeArgumentTable(descriptor: argumentTableDescriptor)
            argumentTable.setAddress(dataBuffer.gpuAddress, index: 0)
            argumentTable.setAddress(mergeSizeBuffer.gpuAddress, index: 1)
            argumentTable.setAddress(strideBuffer.gpuAddress, index: 2)
            
            computeCommandEncoder.setArgumentTable(argumentTable)
            computeCommandEncoder.barrier(afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch)
            
            computeCommandEncoder.dispatchThreads(
                threadsPerGrid: MTLSizeMake(data.count/2, 1, 1),
                threadsPerThreadgroup: MTLSizeMake(computePipelineState.maxTotalThreadsPerThreadgroup, 1, 1))
            
            stride /= 2
        }
        mergeSize *= 2
    }
    
    computeCommandEncoder.endEncoding()
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
    
    return data
}

private func mergeSortBitonic(data: inout [Int32]) async -> [Int32] {
    let device = MTLCreateSystemDefaultDevice()!
    let library = try! device.makeDefaultLibrary(bundle: .main)
    
    let dataBuffer = device.makeBuffer(bytesNoCopy: &data, length: MemoryLayout<Int32>.stride * data.count, options: .storageModeShared)!
    var length = UInt32(data.count)
    let lengthBuffer = device.makeBuffer(bytesNoCopy: &length, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    
    let residencySet = try! device.makeResidencySet(descriptor: .init())
    residencySet.addAllocation(dataBuffer)
    residencySet.addAllocation(lengthBuffer)
    residencySet.commit()
    
    let commandAllocator = device.makeCommandAllocator()!
    let commandBuffer = device.makeCommandBuffer()!
    commandBuffer.beginCommandBuffer(allocator: commandAllocator, options: .init())
    
    let kernelFunction = library.makeFunction(name: "mergeSortBitonic")!
    let computePipelineState = try! await device.makeComputePipelineState(function: kernelFunction)
    let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoder.setComputePipelineState(computePipelineState)
    let argumentTableDescriptor = MTL4ArgumentTableDescriptor()
    argumentTableDescriptor.maxBufferBindCount = 4
    
    var mergeSize: UInt32 = 2
    while mergeSize <= data.count {
        var stride = mergeSize / 2
        while stride > 0 {
            let mergeSizeBuffer = device.makeBuffer(bytes: &mergeSize, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
            let strideBuffer = device.makeBuffer(bytes: &stride, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
            residencySet.addAllocation(mergeSizeBuffer)
            residencySet.addAllocation(strideBuffer)
            residencySet.commit()
            
            let argumentTable = try! device.makeArgumentTable(descriptor: argumentTableDescriptor)
            argumentTable.setAddress(dataBuffer.gpuAddress, index: 0)
            argumentTable.setAddress(lengthBuffer.gpuAddress, index: 1)
            argumentTable.setAddress(mergeSizeBuffer.gpuAddress, index: 2)
            argumentTable.setAddress(strideBuffer.gpuAddress, index: 3)
            
            computeCommandEncoder.setArgumentTable(argumentTable)
            computeCommandEncoder.barrier(afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch)
            
            computeCommandEncoder.dispatchThreads(
                threadsPerGrid: MTLSizeMake(data.count/2, 1, 1),
                threadsPerThreadgroup: MTLSizeMake(computePipelineState.maxTotalThreadsPerThreadgroup, 1, 1))
            
            stride /= 2
        }
        mergeSize <<= 1
    }
    
    computeCommandEncoder.endEncoding()
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
    
    return data
}

private func mergeSortBitonicThreadgroup(data: inout [Int32]) async -> [Int32] {
    let device = MTLCreateSystemDefaultDevice()!
    let library = try! device.makeDefaultLibrary(bundle: .main)
    
    let dataBuffer = device.makeBuffer(bytesNoCopy: &data, length: MemoryLayout<Int32>.stride * data.count, options: .storageModeShared)!
    var length = UInt32(data.count)
    let lengthBuffer = device.makeBuffer(bytesNoCopy: &length, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    
    let residencySet = try! device.makeResidencySet(descriptor: .init())
    residencySet.addAllocation(dataBuffer)
    residencySet.addAllocation(lengthBuffer)
    residencySet.commit()
    
    let commandAllocator = device.makeCommandAllocator()!
    let commandBuffer = device.makeCommandBuffer()!
    commandBuffer.beginCommandBuffer(allocator: commandAllocator, options: .init())
    
    let kernelFunctionThreadgroup = library.makeFunction(name: "mergeSortBitonicThreadgroup")!
    let computePipelineStateThreadgroup = try! await device.makeComputePipelineState(function: kernelFunctionThreadgroup)
    let computeCommandEncoderThreadgroup = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoderThreadgroup.setComputePipelineState(computePipelineStateThreadgroup)
    
    let argumentTableDescriptorThreadgroup = MTL4ArgumentTableDescriptor()
    argumentTableDescriptorThreadgroup.maxBufferBindCount = 2
    let argumentTableThreadgroup = try! device.makeArgumentTable(descriptor: argumentTableDescriptorThreadgroup)
    argumentTableThreadgroup.setAddress(dataBuffer.gpuAddress, index: 0)
    argumentTableThreadgroup.setAddress(lengthBuffer.gpuAddress, index: 1)
    computeCommandEncoderThreadgroup.setArgumentTable(argumentTableThreadgroup)
    computeCommandEncoderThreadgroup.dispatchThreads(
        threadsPerGrid: MTLSizeMake(data.count/2, 1, 1),
        threadsPerThreadgroup: MTLSizeMake(1024, 1, 1))
    computeCommandEncoderThreadgroup.barrier(afterStages: .dispatch, beforeQueueStages: .dispatch)
    computeCommandEncoderThreadgroup.endEncoding()
    
    let kernelFunction = library.makeFunction(name: "mergeSortBitonic")!
    let computePipelineState = try! await device.makeComputePipelineState(function: kernelFunction)
    let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoder.setComputePipelineState(computePipelineState)
    let argumentTableDescriptor = MTL4ArgumentTableDescriptor()
    argumentTableDescriptor.maxBufferBindCount = 4
    
    var mergeSize: UInt32 = 2048
    while mergeSize <= data.count {
        var stride = mergeSize / 2
        while stride > 0 {
            let mergeSizeBuffer = device.makeBuffer(bytes: &mergeSize, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
            let strideBuffer = device.makeBuffer(bytes: &stride, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
            residencySet.addAllocation(mergeSizeBuffer)
            residencySet.addAllocation(strideBuffer)
            residencySet.commit()
            
            let argumentTable = try! device.makeArgumentTable(descriptor: argumentTableDescriptor)
            argumentTable.setAddress(dataBuffer.gpuAddress, index: 0)
            argumentTable.setAddress(lengthBuffer.gpuAddress, index: 1)
            argumentTable.setAddress(mergeSizeBuffer.gpuAddress, index: 2)
            argumentTable.setAddress(strideBuffer.gpuAddress, index: 3)
            
            computeCommandEncoder.setArgumentTable(argumentTable)
            computeCommandEncoder.barrier(afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch)
            
            computeCommandEncoder.dispatchThreads(
                threadsPerGrid: MTLSizeMake(data.count/2, 1, 1),
                threadsPerThreadgroup: MTLSizeMake(computePipelineState.maxTotalThreadsPerThreadgroup, 1, 1))
            
            stride /= 2
        }
        mergeSize *= 2
    }
    
    computeCommandEncoder.endEncoding()
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
    
    return data
}

func metalSort() async {
    let lengths = [
        Int(pow(2.0, 15)),
        Int(pow(2.0, 18)),
        Int(pow(2.0, 20)),
        Int(pow(2.0, 25)),
    ]
    
    for length in lengths {
        var data = await ParallelData.randomInt32Array(length: length)
        var startTime = CFAbsoluteTimeGetCurrent()
        var results = await mergeSortOddEven(data: &data)
        print(String(format: "%d mergeSortOddEven elapsed time: %f seconds\n", length, CFloat(CFAbsoluteTimeGetCurrent() - startTime)))
        checkSorted(results)
        
        data = await ParallelData.randomInt32Array(length: length)
        startTime = CFAbsoluteTimeGetCurrent()
        results = await mergeSortOddEvenThreadgroup(data: &data)
        print(String(format: "%d mergeSortOddEvenThreadgroup elapsed time: %f seconds\n", length, CFloat(CFAbsoluteTimeGetCurrent() - startTime)))
        checkSorted(results)
        
        data = await ParallelData.randomInt32Array(length: length)
        startTime = CFAbsoluteTimeGetCurrent()
        results = await mergeSortBitonic(data: &data)
        print(String(format: "%d mergeSortBitonic elapsed time: %f seconds\n", length, CFloat(CFAbsoluteTimeGetCurrent() - startTime)))
        checkSorted(results)
        
        data = await ParallelData.randomInt32Array(length: length)
        startTime = CFAbsoluteTimeGetCurrent()
        results = await mergeSortBitonicThreadgroup(data: &data)
        print(String(format: "%d mergeSortBitonicThreadgroup elapsed time: %f seconds\n", length, CFloat(CFAbsoluteTimeGetCurrent() - startTime)))
        checkSorted(results)
        
        data = await ParallelData.randomInt32Array(length: length)
        startTime = CFAbsoluteTimeGetCurrent()
        data.sort()
        print(String(format: "%d built-in sort elapsed time: %f seconds\n", length, CFloat(CFAbsoluteTimeGetCurrent() - startTime)))
        checkSorted(data)
    }
}
