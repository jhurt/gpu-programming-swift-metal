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

private func sumThreadgroupBarrier(inputLength: Int) async {
    let device = MTLCreateSystemDefaultDevice()!
    
    var input = [Int32](repeating: 0, count: inputLength)
    for i in 0..<input.count {
        input[i] = Int32(i) + 1
    }
    
    let threadsPerThreadgroup = MTLSizeMake(256, 1, 1)
    
    var output = [Int32](repeating: 0, count: inputLength)
    let inputBuffer = device.makeBuffer(bytesNoCopy: &input, length: MemoryLayout<Int32>.stride * input.count, options: .storageModeShared)!
    let outputBuffer = device.makeBuffer(bytesNoCopy: &output, length: MemoryLayout<Int32>.stride * (input.count / threadsPerThreadgroup.width), options: .storageModeShared)!
    let commandBuffer = device.makeCommandBuffer()!
    
    let commandBufferOptions = MTL4CommandBufferOptions()
    let logStateDescriptor = MTLLogStateDescriptor()
    logStateDescriptor.level = .info
    logStateDescriptor.bufferSize = 1024 * 1024
    let logState = try! device.makeLogState(descriptor: logStateDescriptor)
    logState.addLogHandler { (subsystem, category, level, message) in
        print("\(message)")
    }
    commandBufferOptions.logState = logState
    
    let argumentTableDescriptor = MTL4ArgumentTableDescriptor()
    argumentTableDescriptor.maxBufferBindCount = 2
    let argumentTable = try! device.makeArgumentTable(descriptor: argumentTableDescriptor)
    argumentTable.setAddress(inputBuffer.gpuAddress, index: 0)
    argumentTable.setAddress(outputBuffer.gpuAddress, index: 1)
    
    let residencySet = try! device.makeResidencySet(descriptor: .init())
    residencySet.addAllocation(inputBuffer)
    residencySet.addAllocation(outputBuffer)
    residencySet.commit()
    
    let commandAllocator = device.makeCommandAllocator()!
    
    commandBuffer.beginCommandBuffer(allocator: commandAllocator, options: commandBufferOptions)
    
    let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
    
    let library = try! device.makeDefaultLibrary(bundle: .main)
    let kernelFunction = library.makeFunction(name: "sumThreadgroupBarrier")!
    
    let computePipelineState = try! await device.makeComputePipelineState(function: kernelFunction)
    computeCommandEncoder.setComputePipelineState(computePipelineState)
    computeCommandEncoder.setArgumentTable(argumentTable)
    
    let threadsPerGrid = MTLSizeMake(input.count, 1, 1)
    computeCommandEncoder.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    
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
    
    for i in 0..<input.count / threadsPerThreadgroup.width {
        print("Output[\(i)] = \(output[i])")
    }
}

private func syncNoFence(inputLength: Int) async {
    let device = MTLCreateSystemDefaultDevice()!
    
    var input = [Int32](repeating: 0, count: inputLength)
    for i in 0..<input.count {
        input[i] = Int32(i) + 1
    }
    var output = [Int32](repeating: 0, count: inputLength)
    let inputBuffer = device.makeBuffer(bytesNoCopy: &input, length: MemoryLayout<Int32>.stride * input.count, options: .storageModeShared)!
    // (1) create an intermediate buffer to be used for output on the first pass and input on the second pass
    let intermediateBuffer = device.makeBuffer(length: MemoryLayout<Int32>.stride * input.count, options: .storageModeShared)!
    let outputBuffer = device.makeBuffer(bytesNoCopy: &output, length: MemoryLayout<Int32>.stride * input.count, options: .storageModeShared)!
    let commandBuffer = device.makeCommandBuffer()!
    
    let commandBufferOptions = MTL4CommandBufferOptions()
    let logStateDescriptor = MTLLogStateDescriptor()
    logStateDescriptor.level = .info
    logStateDescriptor.bufferSize = 1024 * 1024
    let logState = try! device.makeLogState(descriptor: logStateDescriptor)
    logState.addLogHandler { (subsystem, category, level, message) in
        print("\(message)")
    }
    commandBufferOptions.logState = logState
    
    let argumentTableDescriptor = MTL4ArgumentTableDescriptor()
    argumentTableDescriptor.maxBufferBindCount = 2
    
    // (2) create an argument table for the first pass
    let argumentTable1 = try! device.makeArgumentTable(descriptor: argumentTableDescriptor)
    argumentTable1.setAddress(inputBuffer.gpuAddress, index: 0)
    argumentTable1.setAddress(intermediateBuffer.gpuAddress, index: 1)
    
    // (3) create an argument table for the first pass
    let argumentTable2 = try! device.makeArgumentTable(descriptor: argumentTableDescriptor)
    argumentTable2.setAddress(intermediateBuffer.gpuAddress, index: 0)
    argumentTable2.setAddress(outputBuffer.gpuAddress, index: 1)
    
    let residencySet = try! device.makeResidencySet(descriptor: .init())
    residencySet.addAllocation(inputBuffer)
    residencySet.addAllocation(intermediateBuffer)
    residencySet.addAllocation(outputBuffer)
    residencySet.commit()
    
    let commandAllocator = device.makeCommandAllocator()!
    
    commandBuffer.beginCommandBuffer(allocator: commandAllocator, options: commandBufferOptions)
    
    let library = try! device.makeDefaultLibrary(bundle: .main)
    let kernelFunction = library.makeFunction(name: "memoryOutputInc")!
    let computePipelineState = try! await device.makeComputePipelineState(function: kernelFunction)
    
    let threadsPerThreadgroup = MTLSizeMake(computePipelineState.threadExecutionWidth, 1, 1)
    let threadsPerGrid = MTLSizeMake(input.count, 1, 1)
    
    // (4) encode a compute command for the first pass
    let computeCommandEncoder1 = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoder1.setComputePipelineState(computePipelineState)
    computeCommandEncoder1.setArgumentTable(argumentTable1)
    computeCommandEncoder1.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    computeCommandEncoder1.endEncoding()
    
    // (5) encode a compute command for the second pass
    let computeCommandEncoder2 = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoder2.setComputePipelineState(computePipelineState)
    computeCommandEncoder2.setArgumentTable(argumentTable2)
    computeCommandEncoder2.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    computeCommandEncoder2.endEncoding()
    
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
    
    for i in 0..<output.count {
        print("Output[\(i)] = \(output[i])")
    }
}

private func syncFence(inputLength: Int) async {
    let device = MTLCreateSystemDefaultDevice()!
    
    var input = [Int32](repeating: 0, count: inputLength)
    for i in 0..<input.count {
        input[i] = Int32(i) + 1
    }
    var output = [Int32](repeating: 0, count: inputLength)
    let inputBuffer = device.makeBuffer(bytesNoCopy: &input, length: MemoryLayout<Int32>.stride * input.count, options: .storageModeShared)!
    let intermediateBuffer = device.makeBuffer(length: MemoryLayout<Int32>.stride * input.count, options: .storageModeShared)!
    let outputBuffer = device.makeBuffer(bytesNoCopy: &output, length: MemoryLayout<Int32>.stride * input.count, options: .storageModeShared)!
    let commandBuffer = device.makeCommandBuffer()!
    
    let commandBufferOptions = MTL4CommandBufferOptions()
    let logStateDescriptor = MTLLogStateDescriptor()
    logStateDescriptor.level = .info
    logStateDescriptor.bufferSize = 1024 * 1024
    let logState = try! device.makeLogState(descriptor: logStateDescriptor)
    logState.addLogHandler { (subsystem, category, level, message) in
        print("\(message)")
    }
    commandBufferOptions.logState = logState
    
    let argumentTableDescriptor = MTL4ArgumentTableDescriptor()
    argumentTableDescriptor.maxBufferBindCount = 2
    
    let argumentTable1 = try! device.makeArgumentTable(descriptor: argumentTableDescriptor)
    argumentTable1.setAddress(inputBuffer.gpuAddress, index: 0)
    argumentTable1.setAddress(intermediateBuffer.gpuAddress, index: 1)
    
    let argumentTable2 = try! device.makeArgumentTable(descriptor: argumentTableDescriptor)
    argumentTable2.setAddress(intermediateBuffer.gpuAddress, index: 0)
    argumentTable2.setAddress(outputBuffer.gpuAddress, index: 1)
    
    let residencySet = try! device.makeResidencySet(descriptor: .init())
    residencySet.addAllocation(inputBuffer)
    residencySet.addAllocation(intermediateBuffer)
    residencySet.addAllocation(outputBuffer)
    residencySet.commit()
    
    let commandAllocator = device.makeCommandAllocator()!
    
    // (1) create an `MTLFence` instance using `makeFence` on the device.
    let fence: MTLFence = device.makeFence()!
    commandBuffer.beginCommandBuffer(allocator: commandAllocator, options: commandBufferOptions)
    
    let library = try! device.makeDefaultLibrary(bundle: .main)
    let kernelFunction = library.makeFunction(name: "memoryOutputInc")!
    let computePipelineState = try! await device.makeComputePipelineState(function: kernelFunction)
    
    let threadsPerThreadgroup = MTLSizeMake(computePipelineState.threadExecutionWidth, 1, 1)
    let threadsPerGrid = MTLSizeMake(input.count, 1, 1)
    
    let computeCommandEncoder1 = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoder1.setComputePipelineState(computePipelineState)
    computeCommandEncoder1.setArgumentTable(argumentTable1)
    computeCommandEncoder1.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    // (2) update the fence instance after the dispatch stage is complete
    computeCommandEncoder1.updateFence(fence, afterEncoderStages: [.dispatch])
    computeCommandEncoder1.endEncoding()
    
    let computeCommandEncoder2 = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoder2.setComputePipelineState(computePipelineState)
    computeCommandEncoder2.setArgumentTable(argumentTable2)
    // (3) wait for the fence to be updated from pass one
    computeCommandEncoder2.waitForFence(fence, beforeEncoderStages: [.dispatch])
    computeCommandEncoder2.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    computeCommandEncoder2.endEncoding()
    
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
    
    for i in 0..<output.count {
        print("Output[\(i)] = \(output[i])")
    }
}

private func syncConsumerBarrier(inputLength: Int) async {
    let device = MTLCreateSystemDefaultDevice()!
    
    var input = [Int32](repeating: 0, count: inputLength)
    for i in 0..<input.count {
        input[i] = Int32(i) + 1
    }
    var output = [Int32](repeating: 0, count: inputLength)
    let inputBuffer = device.makeBuffer(bytesNoCopy: &input, length: MemoryLayout<Int32>.stride * input.count, options: .storageModeShared)!
    let intermediateBuffer = device.makeBuffer(length: MemoryLayout<Int32>.stride * input.count, options: .storageModeShared)!
    let outputBuffer = device.makeBuffer(bytesNoCopy: &output, length: MemoryLayout<Int32>.stride * input.count, options: .storageModeShared)!
    let commandBuffer = device.makeCommandBuffer()!
    
    let commandBufferOptions = MTL4CommandBufferOptions()
    let logStateDescriptor = MTLLogStateDescriptor()
    logStateDescriptor.level = .info
    logStateDescriptor.bufferSize = 1024 * 1024
    let logState = try! device.makeLogState(descriptor: logStateDescriptor)
    logState.addLogHandler { (subsystem, category, level, message) in
        print("\(message)")
    }
    commandBufferOptions.logState = logState
    
    let argumentTableDescriptor = MTL4ArgumentTableDescriptor()
    argumentTableDescriptor.maxBufferBindCount = 2
    
    let argumentTable1 = try! device.makeArgumentTable(descriptor: argumentTableDescriptor)
    argumentTable1.setAddress(inputBuffer.gpuAddress, index: 0)
    argumentTable1.setAddress(intermediateBuffer.gpuAddress, index: 1)
    
    let argumentTable2 = try! device.makeArgumentTable(descriptor: argumentTableDescriptor)
    argumentTable2.setAddress(intermediateBuffer.gpuAddress, index: 0)
    argumentTable2.setAddress(outputBuffer.gpuAddress, index: 1)
    
    let residencySet = try! device.makeResidencySet(descriptor: .init())
    residencySet.addAllocation(inputBuffer)
    residencySet.addAllocation(intermediateBuffer)
    residencySet.addAllocation(outputBuffer)
    residencySet.commit()
    
    let commandAllocator = device.makeCommandAllocator()!
    
    commandBuffer.beginCommandBuffer(allocator: commandAllocator, options: commandBufferOptions)
    
    let library = try! device.makeDefaultLibrary(bundle: .main)
    let kernelFunction = library.makeFunction(name: "memoryOutputInc")!
    let computePipelineState = try! await device.makeComputePipelineState(function: kernelFunction)
    
    let threadsPerThreadgroup = MTLSizeMake(computePipelineState.threadExecutionWidth, 1, 1)
    let threadsPerGrid = MTLSizeMake(input.count, 1, 1)
    
    let computeCommandEncoder1 = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoder1.setComputePipelineState(computePipelineState)
    computeCommandEncoder1.setArgumentTable(argumentTable1)
    computeCommandEncoder1.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    computeCommandEncoder1.endEncoding()
    
    let computeCommandEncoder2 = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoder2.setComputePipelineState(computePipelineState)
    computeCommandEncoder2.setArgumentTable(argumentTable2)
    
    // (1) a consumer queue barrier that blocks dispatch stage in pass two from running until dispatch phase in pass one finishes
    computeCommandEncoder2.barrier(afterQueueStages: .dispatch, beforeStages: .dispatch)
    
    computeCommandEncoder2.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    computeCommandEncoder2.endEncoding()
    
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
    
    for i in 0..<output.count {
        print("Output[\(i)] = \(output[i])")
    }
}

private func syncProducerBarrier(inputLength: Int) async {
    let device = MTLCreateSystemDefaultDevice()!
    
    var input = [Int32](repeating: 0, count: inputLength)
    for i in 0..<input.count {
        input[i] = Int32(i) + 1
    }
    var output = [Int32](repeating: 0, count: inputLength)
    let inputBuffer = device.makeBuffer(bytesNoCopy: &input, length: MemoryLayout<Int32>.stride * input.count, options: .storageModeShared)!
    let intermediateBuffer = device.makeBuffer(length: MemoryLayout<Int32>.stride * input.count, options: .storageModeShared)!
    let outputBuffer = device.makeBuffer(bytesNoCopy: &output, length: MemoryLayout<Int32>.stride * input.count, options: .storageModeShared)!
    let commandBuffer = device.makeCommandBuffer()!
    
    let commandBufferOptions = MTL4CommandBufferOptions()
    let logStateDescriptor = MTLLogStateDescriptor()
    logStateDescriptor.level = .info
    logStateDescriptor.bufferSize = 1024 * 1024
    let logState = try! device.makeLogState(descriptor: logStateDescriptor)
    logState.addLogHandler { (subsystem, category, level, message) in
        print("\(message)")
    }
    commandBufferOptions.logState = logState
    
    let argumentTableDescriptor = MTL4ArgumentTableDescriptor()
    argumentTableDescriptor.maxBufferBindCount = 2
    
    let argumentTable1 = try! device.makeArgumentTable(descriptor: argumentTableDescriptor)
    argumentTable1.setAddress(inputBuffer.gpuAddress, index: 0)
    argumentTable1.setAddress(intermediateBuffer.gpuAddress, index: 1)
    
    let argumentTable2 = try! device.makeArgumentTable(descriptor: argumentTableDescriptor)
    argumentTable2.setAddress(intermediateBuffer.gpuAddress, index: 0)
    argumentTable2.setAddress(outputBuffer.gpuAddress, index: 1)
    
    let residencySet = try! device.makeResidencySet(descriptor: .init())
    residencySet.addAllocation(inputBuffer)
    residencySet.addAllocation(intermediateBuffer)
    residencySet.addAllocation(outputBuffer)
    residencySet.commit()
    
    let commandAllocator = device.makeCommandAllocator()!
    
    commandBuffer.beginCommandBuffer(allocator: commandAllocator, options: commandBufferOptions)
    
    let library = try! device.makeDefaultLibrary(bundle: .main)
    let kernelFunction = library.makeFunction(name: "memoryOutputInc")!
    let computePipelineState = try! await device.makeComputePipelineState(function: kernelFunction)
    
    let threadsPerThreadgroup = MTLSizeMake(computePipelineState.threadExecutionWidth, 1, 1)
    let threadsPerGrid = MTLSizeMake(input.count, 1, 1)
    
    let computeCommandEncoder1 = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoder1.setComputePipelineState(computePipelineState)
    computeCommandEncoder1.setArgumentTable(argumentTable1)
    // (1) a producer barrier that blocks dispatch stage in pass two from running until dispatch stage in this pass finishes
    computeCommandEncoder1.barrier(afterStages: .dispatch, beforeQueueStages: .dispatch)
    computeCommandEncoder1.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    computeCommandEncoder1.endEncoding()
    
    let computeCommandEncoder2 = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoder2.setComputePipelineState(computePipelineState)
    computeCommandEncoder2.setArgumentTable(argumentTable2)
    computeCommandEncoder2.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    computeCommandEncoder2.endEncoding()
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
    
    for i in 0..<output.count {
        print("Output[\(i)] = \(output[i])")
    }
}

private func syncIntraPassBarrier(inputLength: Int) async {
    let device = MTLCreateSystemDefaultDevice()!
    
    var input = [Int32](repeating: 0, count: inputLength)
    for i in 0..<input.count {
        input[i] = Int32(i) + 1
    }
    var output = [Int32](repeating: 0, count: inputLength)
    let inputBuffer = device.makeBuffer(bytesNoCopy: &input, length: MemoryLayout<Int32>.stride * input.count, options: .storageModeShared)!
    let intermediateBuffer = device.makeBuffer(length: MemoryLayout<Int32>.stride * input.count, options: .storageModeShared)!
    let outputBuffer = device.makeBuffer(bytesNoCopy: &output, length: MemoryLayout<Int32>.stride * input.count, options: .storageModeShared)!
    let commandBuffer = device.makeCommandBuffer()!
    
    let commandBufferOptions = MTL4CommandBufferOptions()
    let logStateDescriptor = MTLLogStateDescriptor()
    logStateDescriptor.level = .info
    logStateDescriptor.bufferSize = 1024 * 1024
    let logState = try! device.makeLogState(descriptor: logStateDescriptor)
    logState.addLogHandler { (subsystem, category, level, message) in
        print("\(message)")
    }
    commandBufferOptions.logState = logState
    
    let argumentTableDescriptor = MTL4ArgumentTableDescriptor()
    argumentTableDescriptor.maxBufferBindCount = 2
    
    let residencySet = try! device.makeResidencySet(descriptor: .init())
    residencySet.addAllocation(inputBuffer)
    residencySet.addAllocation(intermediateBuffer)
    residencySet.addAllocation(outputBuffer)
    residencySet.commit()
    
    let commandAllocator = device.makeCommandAllocator()!
    
    commandBuffer.beginCommandBuffer(allocator: commandAllocator, options: commandBufferOptions)
    
    let library = try! device.makeDefaultLibrary(bundle: .main)
    let kernelFunction = library.makeFunction(name: "memoryOutputInc")!
    let computePipelineState = try! await device.makeComputePipelineState(function: kernelFunction)
    
    let threadsPerThreadgroup = MTLSizeMake(computePipelineState.threadExecutionWidth, 1, 1)
    let threadsPerGrid = MTLSizeMake(input.count, 1, 1)
    
    let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoder.setComputePipelineState(computePipelineState)

    // (1) first dispatch
    let argumentTable1 = try! device.makeArgumentTable(descriptor: argumentTableDescriptor)
    argumentTable1.setAddress(inputBuffer.gpuAddress, index: 0)
    argumentTable1.setAddress(intermediateBuffer.gpuAddress, index: 1)
    computeCommandEncoder.setArgumentTable(argumentTable1)
    computeCommandEncoder.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)

    // (2) an intra-pass barrier that blocks dispatch stage in pass two from running until dispatch stage in this pass finishes
    computeCommandEncoder.barrier(afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch)

    // (3) second dispatch
    let argumentTable2 = try! device.makeArgumentTable(descriptor: argumentTableDescriptor)
    argumentTable2.setAddress(intermediateBuffer.gpuAddress, index: 0)
    argumentTable2.setAddress(outputBuffer.gpuAddress, index: 1)
    computeCommandEncoder.setArgumentTable(argumentTable2)
    computeCommandEncoder.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    
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
    
    for i in 0..<output.count {
        print("Output[\(i)] = \(output[i])")
    }
}

private func syncEvent(inputLength: Int) async {
    let device = MTLCreateSystemDefaultDevice()!
    
    var input = [Int32](repeating: 0, count: inputLength)
    for i in 0..<input.count {
        input[i] = Int32(i) + 1
    }
    var output = [Int32](repeating: 0, count: inputLength)
    let inputBuffer = device.makeBuffer(bytesNoCopy: &input, length: MemoryLayout<Int32>.stride * input.count, options: .storageModeShared)!
    let intermediateBuffer = device.makeBuffer(length: MemoryLayout<Int32>.stride * input.count, options: .storageModeShared)!
    let outputBuffer = device.makeBuffer(bytesNoCopy: &output, length: MemoryLayout<Int32>.stride * input.count, options: .storageModeShared)!
    
    let library = try! device.makeDefaultLibrary(bundle: .main)
    let kernelFunction = library.makeFunction(name: "memoryOutputInc")!
    let computePipelineState = try! await device.makeComputePipelineState(function: kernelFunction)
    let threadsPerThreadgroup = MTLSizeMake(computePipelineState.threadExecutionWidth, 1, 1)
    let threadsPerGrid = MTLSizeMake(input.count, 1, 1)
    
    let commandBufferOptions = MTL4CommandBufferOptions()
    let logStateDescriptor = MTLLogStateDescriptor()
    logStateDescriptor.level = .info
    logStateDescriptor.bufferSize = 1024 * 1024
    let logState = try! device.makeLogState(descriptor: logStateDescriptor)
    logState.addLogHandler { (subsystem, category, level, message) in
        print("\(message)")
    }
    commandBufferOptions.logState = logState
    
    let argumentTableDescriptor = MTL4ArgumentTableDescriptor()
    argumentTableDescriptor.maxBufferBindCount = 2
    
    let argumentTable1 = try! device.makeArgumentTable(descriptor: argumentTableDescriptor)
    argumentTable1.setAddress(inputBuffer.gpuAddress, index: 0)
    argumentTable1.setAddress(intermediateBuffer.gpuAddress, index: 1)
    
    let argumentTable2 = try! device.makeArgumentTable(descriptor: argumentTableDescriptor)
    argumentTable2.setAddress(intermediateBuffer.gpuAddress, index: 0)
    argumentTable2.setAddress(outputBuffer.gpuAddress, index: 1)
    
    let residencySet = try! device.makeResidencySet(descriptor: .init())
    residencySet.addAllocation(inputBuffer)
    residencySet.addAllocation(intermediateBuffer)
    residencySet.addAllocation(outputBuffer)
    residencySet.commit()
    
    let commandAllocator = device.makeCommandAllocator()!
    
    // (1) encode the first pass onto `commandBuffer1`
    let commandBuffer1 = device.makeCommandBuffer()!
    commandBuffer1.beginCommandBuffer(allocator: commandAllocator, options: commandBufferOptions)
    let computeCommandEncoder1 = commandBuffer1.makeComputeCommandEncoder()!
    computeCommandEncoder1.setComputePipelineState(computePipelineState)
    computeCommandEncoder1.setArgumentTable(argumentTable1)
    computeCommandEncoder1.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    computeCommandEncoder1.endEncoding()
    commandBuffer1.endCommandBuffer()
    
    // (2) encode the second pass onto `commandBuffer2`
    let commandBuffer2 = device.makeCommandBuffer()!
    commandBuffer2.beginCommandBuffer(allocator: commandAllocator, options: commandBufferOptions)
    let computeCommandEncoder2 = commandBuffer2.makeComputeCommandEncoder()!
    computeCommandEncoder2.setComputePipelineState(computePipelineState)
    computeCommandEncoder2.setArgumentTable(argumentTable2)
    computeCommandEncoder2.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    computeCommandEncoder2.endEncoding()
    commandBuffer2.endCommandBuffer()
    
    // (3) create a non-shareable event to synchronize the passes
    let event = device.makeEvent()!
    
    // (4) create a command queue to submit the first pass
    let commandQueue1 = device.makeMTL4CommandQueue()!
    commandQueue1.addResidencySet(residencySet)
    
    // (5) create a command queue to submit the second pass
    let commandQueue2 = device.makeMTL4CommandQueue()!
    commandQueue2.addResidencySet(residencySet)
    
    try! await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
        let commitOptions = MTL4CommitOptions()
        commitOptions.addFeedbackHandler { feedback in
            if let error = feedback.error {
                continuation.resume(throwing: error)
            } else {
                continuation.resume()
            }
        }
        
        // (6) schedule a wait for event value `1` before doing any GPU work
        commandQueue2.waitForEvent(event, value: 1)
        // (7) submit the second pass
        commandQueue2.commit([commandBuffer2])
        
        // (8) submit the first pass
        commandQueue1.commit([commandBuffer1], options: commitOptions)
        // (9) schedule an event that signals `1` after GPU work in `commandBuffer1` is complete
        commandQueue1.signalEvent(event, value: 1)
    }
    
    for i in 0..<output.count {
        print("Output[\(i)] = \(output[i])")
    }
}

private func syncSharedEvent(inputLength: Int) async {
    let device = MTLCreateSystemDefaultDevice()!
    
    var input = [Int32](repeating: 0, count: inputLength)
    for i in 0..<input.count {
        input[i] = Int32(i) + 1
    }
    
    let threadsPerThreadgroup = MTLSizeMake(256, 1, 1)
    
    var output = [Int32](repeating: 0, count: inputLength)
    let inputBuffer = device.makeBuffer(bytesNoCopy: &input, length: MemoryLayout<Int32>.stride * input.count, options: .storageModeShared)!
    let outputBuffer = device.makeBuffer(bytesNoCopy: &output, length: MemoryLayout<Int32>.stride * (input.count / threadsPerThreadgroup.width), options: .storageModeShared)!
    let commandBuffer = device.makeCommandBuffer()!
    
    let commandBufferOptions = MTL4CommandBufferOptions()
    let logStateDescriptor = MTLLogStateDescriptor()
    logStateDescriptor.level = .info
    logStateDescriptor.bufferSize = 1024 * 1024
    let logState = try! device.makeLogState(descriptor: logStateDescriptor)
    logState.addLogHandler { (subsystem, category, level, message) in
        print("\(message)")
    }
    commandBufferOptions.logState = logState
    
    let argumentTableDescriptor = MTL4ArgumentTableDescriptor()
    argumentTableDescriptor.maxBufferBindCount = 2
    let argumentTable = try! device.makeArgumentTable(descriptor: argumentTableDescriptor)
    argumentTable.setAddress(inputBuffer.gpuAddress, index: 0)
    argumentTable.setAddress(outputBuffer.gpuAddress, index: 1)
    
    let residencySet = try! device.makeResidencySet(descriptor: .init())
    residencySet.addAllocation(inputBuffer)
    residencySet.addAllocation(outputBuffer)
    residencySet.commit()
    
    let commandAllocator = device.makeCommandAllocator()!
    
    commandBuffer.beginCommandBuffer(allocator: commandAllocator, options: commandBufferOptions)
    
    let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
    
    let library = try! device.makeDefaultLibrary(bundle: .main)
    let kernelFunction = library.makeFunction(name: "sumThreadgroupBarrier")!
    
    let computePipelineState = try! await device.makeComputePipelineState(function: kernelFunction)
    computeCommandEncoder.setComputePipelineState(computePipelineState)
    computeCommandEncoder.setArgumentTable(argumentTable)
    
    let threadsPerGrid = MTLSizeMake(input.count, 1, 1)
    computeCommandEncoder.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    
    computeCommandEncoder.endEncoding()
    
    commandBuffer.endCommandBuffer()
    
    let commandQueue = device.makeMTL4CommandQueue()!
    commandQueue.addResidencySet(residencySet)

    // (1) create a shared event to synchronize the CPU and GPU
    let sharedEvent = device.makeSharedEvent()!
    
    // (2) commit the work in `commandBuffer` to execute on the GPU
    commandQueue.commit([commandBuffer])
    
    // (3) schedule an event that signals `1` after GPU work in `commandBuffer` is complete
    commandQueue.signalEvent(sharedEvent, value: 1)

    // (4) block the CPU on the signaling of `1` of the shared event
    await sharedEvent.valueSignaled(1)
    
    for i in 0..<input.count / threadsPerThreadgroup.width {
        print("Output[\(i)] = \(output[i])")
    }
}

func metalThreadCoordination() async {
    print("Sum Threadgroup Barrier")
    await sumThreadgroupBarrier(inputLength: 4096)
    
    print("\nSync No Fence")
    await syncNoFence(inputLength: 16)
    
    print("\nSync Fence")
    await syncFence(inputLength: 16)
    
    print("\nSync Consumer Barrier")
    await syncConsumerBarrier(inputLength: 16)
    
    print("\nSync Producer Barrier")
    await syncProducerBarrier(inputLength: 16)
    
    print("\nSync Intra-Pass Barrier")
    await syncIntraPassBarrier(inputLength: 16)
    
    print("\nSync Event")
    await syncEvent(inputLength: 16)
    
    print("\nSync Shared Event")
    await syncSharedEvent(inputLength: 4096)
}
