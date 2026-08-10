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

private func memoryPrintInputCopy(inputLength: Int) async {
    let device = MTLCreateSystemDefaultDevice()!
    
    // (1) create an array of integers
    var input = [Int32](repeating: 0, count: inputLength)
    for i in 0..<input.count {
        input[i] = Int32(i) + 1
    }
    
    let startTime = CFAbsoluteTimeGetCurrent()

    // (2) create a device buffer by copying the array
    let inputBuffer = device.makeBuffer(bytes: &input, length: MemoryLayout<Int32>.stride * input.count, options: .storageModeShared)!
    
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
    
    // (3) create an argument table descriptor to provide resource bindings to the Metal pipeline
    let argumentTableDescriptor = MTL4ArgumentTableDescriptor()

    // (4) set the buffer bind count to 1 since we have one buffer to bind
    argumentTableDescriptor.maxBufferBindCount = 1
    let argumentTable = try! device.makeArgumentTable(descriptor: argumentTableDescriptor)

    // (5) set the buffer address in the argument table at index 0
    argumentTable.setAddress(inputBuffer.gpuAddress, index: 0)

    // (6) create a residency set to make the memory resident (GPU accessible)
    let residencySet = try! device.makeResidencySet(descriptor: .init())
    
    // (7) add the buffer to the residency set
    residencySet.addAllocation(inputBuffer)
    
    // (8) call commit on the residency set any time it changes
    residencySet.commit()
    
    let commandAllocator = device.makeCommandAllocator()!
    
    commandBuffer.beginCommandBuffer(allocator: commandAllocator, options: commandBufferOptions)
    
    let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
    
    let library = try! device.makeDefaultLibrary(bundle: .main)
    let kernelFunction = library.makeFunction(name: "memoryPrintInput")!
    
    let computePipelineState = try! await device.makeComputePipelineState(function: kernelFunction)
    computeCommandEncoder.setComputePipelineState(computePipelineState)
    
    // (9) set the argument table on the compute command encoder
    computeCommandEncoder.setArgumentTable(argumentTable)
    
    let threadsPerThreadgroup = MTLSizeMake(computePipelineState.maxTotalThreadsPerThreadgroup, 1, 1)
    
    // (10) set the thread count to the size of the buffer, one thread per element
    let threadsPerGrid = MTLSizeMake(input.count, 1, 1)
    
    computeCommandEncoder.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    
    computeCommandEncoder.endEncoding()
    
    commandBuffer.endCommandBuffer()
    
    let commandQueue = device.makeMTL4CommandQueue()!
    
    // (11) associate the residency set with the command queue
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
    
    print("memoryPrintInputCopy: \(CFAbsoluteTimeGetCurrent() - startTime) seconds")
}

private func memoryPrintInputNoCopy(inputLength: Int) async {
    let device = MTLCreateSystemDefaultDevice()!
    
    var input = [Int32](repeating: 0, count: inputLength)
    for i in 0..<input.count {
        input[i] = Int32(i) + 1
    }
    
    let startTime = CFAbsoluteTimeGetCurrent()
    // (1) create a device buffer by without copying the array
    let inputBuffer = device.makeBuffer(bytesNoCopy: &input, length: MemoryLayout<Int32>.stride * input.count, options: .storageModeShared)!
    
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
    argumentTableDescriptor.maxBufferBindCount = 1
    let argumentTable = try! device.makeArgumentTable(descriptor: argumentTableDescriptor)
    argumentTable.setAddress(inputBuffer.gpuAddress, index: 0)

    let residencySet = try! device.makeResidencySet(descriptor: .init())
    residencySet.addAllocation(inputBuffer)
    residencySet.commit()
    
    let commandAllocator = device.makeCommandAllocator()!
    
    commandBuffer.beginCommandBuffer(allocator: commandAllocator, options: commandBufferOptions)
    
    let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
    
    let library = try! device.makeDefaultLibrary(bundle: .main)
    let kernelFunction = library.makeFunction(name: "memoryPrintInput")!
    
    let computePipelineState = try! await device.makeComputePipelineState(function: kernelFunction)
    computeCommandEncoder.setComputePipelineState(computePipelineState)
    computeCommandEncoder.setArgumentTable(argumentTable)
    
    let threadsPerThreadgroup = MTLSizeMake(computePipelineState.maxTotalThreadsPerThreadgroup, 1, 1)
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
    
    print("memoryPrintInputNoCopy: \(CFAbsoluteTimeGetCurrent() - startTime) seconds")
}

private func memoryOutputIncGPUAlloc(inputLength: Int) async {
    let device = MTLCreateSystemDefaultDevice()!
    
    var input = [Int32](repeating: 0, count: inputLength)
    for i in 0..<input.count {
        input[i] = Int32(i) + 1
    }
    let inputBuffer = device.makeBuffer(bytes: &input, length: MemoryLayout<Int32>.stride * input.count, options: .storageModeShared)!
    
    // (1) create a device output buffer on the GPU
    let outputBuffer = device.makeBuffer(length: MemoryLayout<Int32>.stride * input.count, options: .storageModeShared)!
    
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
    let kernelFunction = library.makeFunction(name: "memoryOutputInc")!
    
    let computePipelineState = try! await device.makeComputePipelineState(function: kernelFunction)
    computeCommandEncoder.setComputePipelineState(computePipelineState)
    computeCommandEncoder.setArgumentTable(argumentTable)
    
    let threadsPerThreadgroup = MTLSizeMake(computePipelineState.maxTotalThreadsPerThreadgroup, 1, 1)
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
    
    // (2) bind the output buffer memory to a Swift `UnsafeBufferPointer`
    let outputPtr = outputBuffer.contents().bindMemory(to: Int32.self, capacity: input.count)
    let bufferPointer = UnsafeBufferPointer(start: outputPtr, count: input.count)

    // (3) wrap the pointer in an array and print the results
    let array = Array(bufferPointer)
    for i in 0..<array.count {
        print("Output[\(i)] = \(array[i])")
    }
}

private func memoryOutputIncHostAlloc(inputLength: Int) async {
    let device = MTLCreateSystemDefaultDevice()!
    
    var input = [Int32](repeating: 0, count: inputLength)
    for i in 0..<input.count {
        input[i] = Int32(i) + 1
    }
    let inputBuffer = device.makeBuffer(bytesNoCopy: &input, length: MemoryLayout<Int32>.stride * input.count, options: .storageModeShared)!

    // (1) create memory on the host
    var output = [Int32](repeating: 0, count: inputLength)
    
    // (2) pass a pointer to it for access on the GPU
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
    let kernelFunction = library.makeFunction(name: "memoryOutputInc")!
    
    let computePipelineState = try! await device.makeComputePipelineState(function: kernelFunction)
    computeCommandEncoder.setComputePipelineState(computePipelineState)
    computeCommandEncoder.setArgumentTable(argumentTable)
    
    let threadsPerThreadgroup = MTLSizeMake(computePipelineState.maxTotalThreadsPerThreadgroup, 1, 1)
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
    
    // (3) print the output array results
    for i in 0..<output.count {
        print("Output[\(i)] = \(output[i])")
    }
}

private func memoryOutputIncThreadgroupMemory(inputLength: Int) async {
    let device = MTLCreateSystemDefaultDevice()!
    
    var input = [Int32](repeating: 0, count: inputLength)
    for i in 0..<input.count {
        input[i] = Int32(i) + 1
    }
    var output = [Int32](repeating: 0, count: inputLength)
    let inputBuffer = device.makeBuffer(bytesNoCopy: &input, length: MemoryLayout<Int32>.stride * input.count, options: .storageModeShared)!
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
    let kernelFunction = library.makeFunction(name: "memoryOutputIncThreadgroupMemory")!
    
    let computePipelineState = try! await device.makeComputePipelineState(function: kernelFunction)
    computeCommandEncoder.setComputePipelineState(computePipelineState)
    computeCommandEncoder.setArgumentTable(argumentTable)
    // (1) calculate the threadgroup memory length needed
    let threadgroupMemoryLength = MemoryLayout<Int32>.stride * input.count
    // (2) allocate threadgroup memory for the compute kernel at index 0
    computeCommandEncoder.setThreadgroupMemoryLength(threadgroupMemoryLength, index: 0)
    
    let threadsPerThreadgroup = MTLSizeMake(computePipelineState.maxTotalThreadsPerThreadgroup, 1, 1)
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
    
    for i in 0..<output.count {
        print("Output[\(i)] = \(output[i])")
    }
}

private func memoryOutputIncConstant(inputLength: Int) async {
    let device = MTLCreateSystemDefaultDevice()!
    
    var input = [Int32](repeating: 0, count: inputLength)
    for i in 0..<input.count {
        input[i] = Int32(i) + 1
    }
    var output = [Int32](repeating: 0, count: inputLength)
    let inputBuffer = device.makeBuffer(bytesNoCopy: &input, length: MemoryLayout<Int32>.stride * input.count, options: .storageModeShared)!
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
    let kernelFunction = library.makeFunction(name: "memoryOutputIncConstant")!
    
    let computePipelineState = try! await device.makeComputePipelineState(function: kernelFunction)
    computeCommandEncoder.setComputePipelineState(computePipelineState)
    computeCommandEncoder.setArgumentTable(argumentTable)
    
    let threadsPerThreadgroup = MTLSizeMake(computePipelineState.maxTotalThreadsPerThreadgroup, 1, 1)
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
    
    for i in 0..<output.count {
        print("Output[\(i)] = \(output[i])")
    }
}

private func memoryOutputIncFunctionConstant(inputLength: Int) async {
    let device = MTLCreateSystemDefaultDevice()!
    
    var input = [Int32](repeating: 0, count: inputLength)
    for i in 0..<input.count {
        input[i] = Int32(i) + 1
    }
    var output = [Int32](repeating: 0, count: inputLength)
    let inputBuffer = device.makeBuffer(bytesNoCopy: &input, length: MemoryLayout<Int32>.stride * input.count, options: .storageModeShared)!
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
    // (1) create function constant values
    let functionConstantValues = MTLFunctionConstantValues()
    var functionConstantIncrementValue = 100
    functionConstantValues.setConstantValue(&functionConstantIncrementValue, type: .int, index: 0)
    // (2) pass the values when creating the kernel function
    let kernelFunction = try! await library.makeFunction(name: "memoryOutputIncFunctionConstant", constantValues: functionConstantValues)
    
    let computePipelineState = try! await device.makeComputePipelineState(function: kernelFunction)
    computeCommandEncoder.setComputePipelineState(computePipelineState)
    computeCommandEncoder.setArgumentTable(argumentTable)
    
    let threadsPerThreadgroup = MTLSizeMake(computePipelineState.maxTotalThreadsPerThreadgroup, 1, 1)
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
    
    for i in 0..<output.count {
        print("Output[\(i)] = \(output[i])")
    }
}

private func memoryHeap(inputLength: Int) async {
    let device = MTLCreateSystemDefaultDevice()!
    
    var input = [Int32](repeating: 0, count: inputLength)
    for i in 0..<input.count {
        input[i] = Int32(i) + 1
    }
    let inputBuffer = device.makeBuffer(bytesNoCopy: &input, length: MemoryLayout<Int32>.stride * input.count, options: .storageModeShared)!
    
    // (1) create an `MTLHeapDescriptor`
    let heapDescriptor = MTLHeapDescriptor()
    // (2) specify a size large enough to hold both output buffers
    heapDescriptor.size = MemoryLayout<Int32>.stride * input.count * 2
    // (3) set `storageModeShared` as the storage mode
    heapDescriptor.storageMode = .shared
    // (4) create the heap using `makeHeap` on the device
    let heap = device.makeHeap(descriptor: heapDescriptor)!
    // (5) allocate the output buffers on the heap
    let outputBuffer1 = heap.makeBuffer(length: MemoryLayout<Int32>.stride  * input.count)!
    let outputBuffer2 = heap.makeBuffer(length: MemoryLayout<Int32>.stride  * input.count)!
    
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
    argumentTableDescriptor.maxBufferBindCount = 3
    let argumentTable = try! device.makeArgumentTable(descriptor: argumentTableDescriptor)
    argumentTable.setAddress(inputBuffer.gpuAddress, index: 0)
    argumentTable.setAddress(outputBuffer1.gpuAddress, index: 1)
    argumentTable.setAddress(outputBuffer2.gpuAddress, index: 2)

    let residencySet = try! device.makeResidencySet(descriptor: .init())
    residencySet.addAllocation(inputBuffer)
    // (6) add the heap to the residency set
    residencySet.addAllocation(heap)
    residencySet.commit()
    
    let commandAllocator = device.makeCommandAllocator()!
    
    commandBuffer.beginCommandBuffer(allocator: commandAllocator, options: commandBufferOptions)
    
    let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
    
    let library = try! device.makeDefaultLibrary(bundle: .main)
    let kernelFunction = library.makeFunction(name: "memoryHeap")!
    
    let computePipelineState = try! await device.makeComputePipelineState(function: kernelFunction)
    computeCommandEncoder.setComputePipelineState(computePipelineState)
    computeCommandEncoder.setArgumentTable(argumentTable)
    
    let threadsPerThreadgroup = MTLSizeMake(computePipelineState.maxTotalThreadsPerThreadgroup, 1, 1)
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
    
    // (7) bind the memory to pointers
    let outputPtr1 = outputBuffer1.contents().bindMemory(to: Int32.self, capacity: input.count)
    let outputPtr2 = outputBuffer2.contents().bindMemory(to: Int32.self, capacity: input.count)
    // (8) print the output array results
    for i in 0..<input.count {
        print("Output1[\(i)] = \(outputPtr1[i]), Output2[\(i)] = \(outputPtr2[i])")
    }
}

private func memoryBandwidth(inputLength: Int) async {
    let device = MTLCreateSystemDefaultDevice()!
    
    let inputBuffer = device.makeBuffer(length: MemoryLayout<SIMD4<Float>>.stride * inputLength, options: .storageModeShared)!
    let outputBuffer = device.makeBuffer(length: MemoryLayout<SIMD4<Float>>.stride * inputLength, options: .storageModeShared)!
    let commandBuffer = device.makeCommandBuffer()!
    
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
    commandBuffer.beginCommandBuffer(allocator: commandAllocator)
    let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
    
    let library = try! device.makeDefaultLibrary(bundle: .main)
    let kernelFunction = library.makeFunction(name: "memoryBandwidth")!
    let computePipelineState = try! await device.makeComputePipelineState(function: kernelFunction)
    computeCommandEncoder.setComputePipelineState(computePipelineState)
    computeCommandEncoder.setArgumentTable(argumentTable)

    let threadsPerThreadgroup = MTLSizeMake(computePipelineState.maxTotalThreadsPerThreadgroup, 1, 1)
    let threadsPerGrid = MTLSizeMake(inputLength, 1, 1)
    let dispatchCount = 200
    for _ in 0..<dispatchCount {
        computeCommandEncoder.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    }
    computeCommandEncoder.endEncoding()
    
    commandBuffer.endCommandBuffer()
    
    let commandQueue = device.makeMTL4CommandQueue()!
    commandQueue.addResidencySet(residencySet)
    let commitOptions = MTL4CommitOptions()
    
    // (1) total byte count is multiplied by two, one for read and one for write
    let totalBytes = inputBuffer.length * dispatchCount * 2
    try! await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
        commitOptions.addFeedbackHandler { feedback in
            if let error = feedback.error {
                continuation.resume(throwing: error)
            } else {
                continuation.resume()
            }
            
            // (2) get the GPU time reported by the MTL4CommitFeedbackHandler
            let gpuTime = feedback.gpuEndTime - feedback.gpuStartTime
            print("GPU Time: \(String(format: "%.4f", gpuTime)) seconds")
            
            // (3) calculate and report throughput
            let gigabytes = Double(totalBytes) / (1024.0 * 1024.0 * 1024.0)
            let bandwidth = gigabytes / gpuTime
            print("Throughput: \(String(format: "%.2f", bandwidth)) GB/s")
        }
        commandQueue.commit([commandBuffer], options: commitOptions)
    }
}

private func memoryLimits() {
    let device = MTLCreateSystemDefaultDevice()!

    let maxWorkingSetMB = Double(device.recommendedMaxWorkingSetSize) / (1024.0 * 1024.0)
    print(String(format: "Recommended max working set memory: %.1f MB", maxWorkingSetMB))

    let maxThreadgroupMemoryKB = Double(device.maxThreadgroupMemoryLength) / 1024.0
    print(String(format:"Max threadgroup memory: %.1f KB", maxThreadgroupMemoryKB))
}

private func memoryRandom(threadCount: Int, chunkLength: Int, kernelName: String, shuffled: Bool) async {
    let device = MTLCreateSystemDefaultDevice()!
    
    // (1) create array of random floats, each thread will read a single chunk of data
    var input = await ParallelData.randomFloatArray(length: threadCount * chunkLength)

    // (2) create array of indices into the input and shuffle if specified
    var indices: [Int] = Array(0..<input.count)
    if shuffled {
        indices.shuffle()
    }
    
    let inputBuffer = device.makeBuffer(bytes: &input, length: MemoryLayout<Float32>.stride * input.count, options: .storageModeShared)!
    let indicesBuffer = device.makeBuffer(bytes: &indices, length: MemoryLayout<Int>.stride * indices.count, options: .storageModeShared)!
    var chunkLengthUInt32 = UInt32(chunkLength)
    let chunkLengthBuffer = device.makeBuffer(bytes: &chunkLengthUInt32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    let outputBuffer = device.makeBuffer(length: MemoryLayout<Float32>.stride * input.count, options: .storageModeShared)!
    
    let argumentTableDescriptor = MTL4ArgumentTableDescriptor()
    argumentTableDescriptor.maxBufferBindCount = 4
    
    let argumentTable = try! device.makeArgumentTable(descriptor: argumentTableDescriptor)
    argumentTable.setAddress(inputBuffer.gpuAddress, index: 0)
    argumentTable.setAddress(indicesBuffer.gpuAddress, index: 1)
    argumentTable.setAddress(chunkLengthBuffer.gpuAddress, index: 2)
    argumentTable.setAddress(outputBuffer.gpuAddress, index: 3)
    
    let residencySet = try! device.makeResidencySet(descriptor: .init())
    residencySet.addAllocation(inputBuffer)
    residencySet.addAllocation(indicesBuffer)
    residencySet.addAllocation(chunkLengthBuffer)
    residencySet.addAllocation(outputBuffer)
    residencySet.commit()
    
    let commandAllocator = device.makeCommandAllocator()!
    
    let library = try! device.makeDefaultLibrary(bundle: .main)
    let kernelFunction = library.makeFunction(name: kernelName)!
    let computePipelineState = try! await device.makeComputePipelineState(function: kernelFunction)
    
    let commandBuffer = device.makeCommandBuffer()!
    commandBuffer.beginCommandBuffer(allocator: commandAllocator, options: .init())
    let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoder.setComputePipelineState(computePipelineState)
    computeCommandEncoder.setArgumentTable(argumentTable)
    computeCommandEncoder.dispatchThreads(threadsPerGrid: MTLSizeMake(threadCount, 1, 1),
                                          threadsPerThreadgroup: MTLSizeMake(computePipelineState.maxTotalThreadsPerThreadgroup, 1, 1))
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
            
            // (3) log the GPU time
            print("memory \(kernelName), \(threadCount) threads, \(MemoryLayout<Float32>.size * chunkLength) byte chunks, \(shuffled ? "shuffled": "contiguous") GPU time: \(feedback.gpuEndTime - feedback.gpuStartTime) seconds\n")
        }
        commandQueue.commit([commandBuffer], options: commitOptions)
    }
}

func metalMemory() async {
    let inputLength = 16
    
    print("memoryPrintInputCopy")
    await memoryPrintInputCopy(inputLength: inputLength)
    
    print("\nmemoryPrintInputNoCopy")
    await memoryPrintInputNoCopy(inputLength: inputLength)
    
    print("\nmemoryOutputIncGPUAlloc")
    await memoryOutputIncGPUAlloc(inputLength: inputLength)
    
    print("\nmemoryOutputIncHostAlloc")
    await memoryOutputIncHostAlloc(inputLength: inputLength)
    
    print("\nmemoryOutputIncThreadgroupMemory")
    await memoryOutputIncThreadgroupMemory(inputLength: inputLength)
    
    print("\nmemoryOutputIncConstant")
    await memoryOutputIncConstant(inputLength: inputLength)
    
    print("\nmemoryOutputIncFunctionConstant")
    await memoryOutputIncFunctionConstant(inputLength: inputLength)
    
    print("\nmemoryHeap")
    await memoryHeap(inputLength: inputLength)
    
    print("\nmemoryBandwidth")
    let bandwidthInputLength = 4_000_000
    await memoryBandwidth(inputLength: bandwidthInputLength)
    
    print("\nmemoryLimits")
    memoryLimits()
    
    let chunkLengths = [1024, 4096]
    let threadCounts = [16384, 131072]
    for threadCount in threadCounts {
        for chunkLength in chunkLengths {
            await memoryRandom(threadCount: threadCount, chunkLength: chunkLength, kernelName: "memoryRandomRead", shuffled: false)
            await memoryRandom(threadCount: threadCount, chunkLength: chunkLength, kernelName: "memoryRandomRead", shuffled: true)
            await memoryRandom(threadCount: threadCount, chunkLength: chunkLength, kernelName: "memoryRandomWrite", shuffled: false)
            await memoryRandom(threadCount: threadCount, chunkLength: chunkLength, kernelName: "memoryRandomWrite", shuffled: true)
            await memoryRandom(threadCount: threadCount, chunkLength: chunkLength, kernelName: "memoryRandomReadWrite", shuffled: true)
        }
    }
}
