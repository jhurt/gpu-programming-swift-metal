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

private func metalGrid1D() async {
    // (1) get a reference to the GPU
    let device = MTLCreateSystemDefaultDevice()!
    
    // (2) create a new command buffer
    let commandBuffer = device.makeCommandBuffer()!
    
    // (3) enable logging for the kernel function
    let commandBufferOptions = MTL4CommandBufferOptions()
    let logStateDescriptor = MTLLogStateDescriptor()
    logStateDescriptor.level = .info
    logStateDescriptor.bufferSize = 1024 * 1024
    let logState = try! device.makeLogState(descriptor: logStateDescriptor)
    logState.addLogHandler { (subsystem, category, level, message) in
        print("\(message)")
    }
    commandBufferOptions.logState = logState
    
    // (4) prepare the command buffer for encoding
    let commandAllocator = device.makeCommandAllocator()!
    commandBuffer.beginCommandBuffer(allocator: commandAllocator, options: commandBufferOptions)
    
    // (5) make a compute command encoder for encoding a compute pass
    let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
    
    // (6) load the "metalGrid1D" kernel function from the default library
    let library = try! device.makeDefaultLibrary(bundle: .main)
    let kernelFunction = library.makeFunction(name: "metalGrid1D")!
    
    // (7) create a MTLComputePipelineState with GPU pipeline configuration for running kernels in a compute pass
    let computePipelineState = try! await device.makeComputePipelineState(function: kernelFunction)
    computeCommandEncoder.setComputePipelineState(computePipelineState)
    
    // (8) define a 1D grid of 16 threads
    let threadsPerGrid = MTLSizeMake(16, 1, 1)
    // all 16 threads belong to a single threadgroup
    let threadsPerThreadgroup = MTLSizeMake(16, 1, 1)
    
    // (9) encode a compute dispatch command
    computeCommandEncoder.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    
    // (10) declares that command generation from this computeCommandEncoder is complete
    computeCommandEncoder.endEncoding()
    
    // (11) close the commandBuffer to prepare it for submission to a command queue
    commandBuffer.endCommandBuffer()
    
    // (12) create a command queue
    let commandQueue = device.makeMTL4CommandQueue()!
    let commitOptions = MTL4CommitOptions()
    
    // (13) use a continuation to wait for the command buffer to complete via a MTL4CommitFeedbackHandler
    try! await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
        commitOptions.addFeedbackHandler { feedback in
            if let error = feedback.error {
                continuation.resume(throwing: error)
            } else {
                continuation.resume()
            }
        }
        
        // (14) commit the command buffer to the GPU
        commandQueue.commit([commandBuffer], options: commitOptions)
    }
}

private func metalGrid2D() async {
    let device = MTLCreateSystemDefaultDevice()!
    
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
    
    let commandAllocator = device.makeCommandAllocator()!
    commandBuffer.beginCommandBuffer(allocator: commandAllocator, options: commandBufferOptions)
    
    let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
    
    let library = try! device.makeDefaultLibrary(bundle: .main)
    let kernelFunction = library.makeFunction(name: "metalGrid2D")!
    
    let computePipelineState = try! await device.makeComputePipelineState(function: kernelFunction)
    computeCommandEncoder.setComputePipelineState(computePipelineState)
    
    // (1) define a 2D 2x8 grid of 16 threads
    let threadsPerGrid = MTLSizeMake(2, 8, 1)
    let threadsPerThreadgroup = MTLSizeMake(2, 8, 1)
    computeCommandEncoder.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    
    computeCommandEncoder.endEncoding()
    
    commandBuffer.endCommandBuffer()
    
    let commandQueue = device.makeMTL4CommandQueue()!
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
}

private func metalGrid3D() async {
    let device = MTLCreateSystemDefaultDevice()!
    
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
    
    let commandAllocator = device.makeCommandAllocator()!
    commandBuffer.beginCommandBuffer(allocator: commandAllocator, options: commandBufferOptions)
    
    let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
    
    let library = try! device.makeDefaultLibrary(bundle: .main)
    let kernelFunction = library.makeFunction(name: "metalGrid3D")!
    
    let computePipelineState = try! await device.makeComputePipelineState(function: kernelFunction)
    computeCommandEncoder.setComputePipelineState(computePipelineState)
    
    // (1) define a 3D 2x2x4 grid of 16 threads
    let threadsPerGrid = MTLSizeMake(2, 2, 4)
    let threadsPerThreadgroup = MTLSizeMake(2, 2, 4)
    computeCommandEncoder.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    
    computeCommandEncoder.endEncoding()
    
    commandBuffer.endCommandBuffer()
    
    let commandQueue = device.makeMTL4CommandQueue()!
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
}

private func metalThreadgroup1D() async {
    let device = MTLCreateSystemDefaultDevice()!
    
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
    
    let commandAllocator = device.makeCommandAllocator()!
    commandBuffer.beginCommandBuffer(allocator: commandAllocator, options: commandBufferOptions)
    
    let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
    
    let library = try! device.makeDefaultLibrary(bundle: .main)
    let kernelFunction = library.makeFunction(name: "metalThreadgroup1D")!
    
    let computePipelineState = try! await device.makeComputePipelineState(function: kernelFunction)
    computeCommandEncoder.setComputePipelineState(computePipelineState)
    
    // (1) define a set of 16 threads to execute in a single 1D threadgroup
    let threadgroupsPerGrid = MTLSizeMake(1, 1, 1)
    let threadsPerThreadgroup = MTLSizeMake(16, 1, 1)
    
    // (2) call dispatchThreadgroups now instead of dispatchThreads
    computeCommandEncoder.dispatchThreadgroups(threadgroupsPerGrid: threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    
    computeCommandEncoder.endEncoding()
    
    commandBuffer.endCommandBuffer()
    
    let commandQueue = device.makeMTL4CommandQueue()!
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
}

private func metalThreadgroup2D() async {
    let device = MTLCreateSystemDefaultDevice()!
    
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
    
    let commandAllocator = device.makeCommandAllocator()!
    commandBuffer.beginCommandBuffer(allocator: commandAllocator, options: commandBufferOptions)
    
    let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
    
    let library = try! device.makeDefaultLibrary(bundle: .main)
    let kernelFunction = library.makeFunction(name: "metalThreadgroup2D")!
    
    let computePipelineState = try! await device.makeComputePipelineState(function: kernelFunction)
    computeCommandEncoder.setComputePipelineState(computePipelineState)
    
    // (1) define a set of 16 threads to execute in a 2D 2x8 threadgroup
    let threadgroupsPerGrid = MTLSizeMake(1, 1, 1)
    let threadsPerThreadgroup = MTLSizeMake(2, 8, 1)
    computeCommandEncoder.dispatchThreadgroups(threadgroupsPerGrid: threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    
    computeCommandEncoder.endEncoding()
    
    commandBuffer.endCommandBuffer()
    
    let commandQueue = device.makeMTL4CommandQueue()!
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
}

private func metalThreadgroup3D() async {
    let device = MTLCreateSystemDefaultDevice()!
    
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
    
    let commandAllocator = device.makeCommandAllocator()!
    commandBuffer.beginCommandBuffer(allocator: commandAllocator, options: commandBufferOptions)
    
    let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
    
    let library = try! device.makeDefaultLibrary(bundle: .main)
    let kernelFunction = library.makeFunction(name: "metalThreadgroup3D")!
    
    let computePipelineState = try! await device.makeComputePipelineState(function: kernelFunction)
    computeCommandEncoder.setComputePipelineState(computePipelineState)
    
    // (1) define a set of 16 threads to execute in a 3D 2x2x4 threadgroup
    let threadgroupsPerGrid = MTLSizeMake(1, 1, 1)
    let threadsPerThreadgroup = MTLSizeMake(2, 2, 4)
    computeCommandEncoder.dispatchThreadgroups(threadgroupsPerGrid: threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    
    computeCommandEncoder.endEncoding()
    
    commandBuffer.endCommandBuffer()
    
    let commandQueue = device.makeMTL4CommandQueue()!
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
}

private func metalThreadgroupSizing1D(gridWidth: Int) async {
    let device = MTLCreateSystemDefaultDevice()!
    
    let commandBuffer = device.makeCommandBuffer()!
    
    let commandBufferOptions = MTL4CommandBufferOptions()
    let logStateDescriptor = MTLLogStateDescriptor()
    logStateDescriptor.level = .info
    logStateDescriptor.bufferSize = 1024 * 1024 * 1024
    let logState = try! device.makeLogState(descriptor: logStateDescriptor)
    var allThreadsPerThreadgroup = Set<String>()
    logState.addLogHandler { (subsystem, category, level, message) in
        let regex = /threadsPerThreadgroup\s+(\d+)/
        if let match = message.firstMatch(of: regex) {
            let threadsPerThreadgroup = String(match.1)
            if !allThreadsPerThreadgroup.contains(threadsPerThreadgroup) {
                print(message)
                allThreadsPerThreadgroup.insert(threadsPerThreadgroup)
            }
        }
    }
    commandBufferOptions.logState = logState
    
    let commandAllocator = device.makeCommandAllocator()!
    commandBuffer.beginCommandBuffer(allocator: commandAllocator, options: commandBufferOptions)
    
    let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
    
    let library = try! device.makeDefaultLibrary(bundle: .main)
    let kernelFunction = library.makeFunction(name: "metalThreadgroup1D")!
    
    let computePipelineState = try! await device.makeComputePipelineState(function: kernelFunction)
    computeCommandEncoder.setComputePipelineState(computePipelineState)
    
    // (1) print max total threads per threadgroup
    print("max total threads per threadgroup: \(computePipelineState.maxTotalThreadsPerThreadgroup)")
    
    // (2) Set the threads per threadgroup to the max total threads per threadgroup
    let threadsPerGrid = MTLSizeMake(gridWidth, 1, 1)
    let threadsPerThreadgroup = MTLSizeMake(computePipelineState.maxTotalThreadsPerThreadgroup, 1, 1)
    computeCommandEncoder.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    
    computeCommandEncoder.endEncoding()
    
    commandBuffer.endCommandBuffer()
    
    let commandQueue = device.makeMTL4CommandQueue()!
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
}

private func metalThreadgroupSizing2D(gridWidth: Int, gridHeight: Int) async {
    let device = MTLCreateSystemDefaultDevice()!
    
    let commandBuffer = device.makeCommandBuffer()!
    
    let commandBufferOptions = MTL4CommandBufferOptions()
    let logStateDescriptor = MTLLogStateDescriptor()
    logStateDescriptor.level = .info
    logStateDescriptor.bufferSize = 1024 * 1024 * 1024
    let logState = try! device.makeLogState(descriptor: logStateDescriptor)
    var allThreadsPerThreadgroup = Set<String>()
    logState.addLogHandler { (subsystem, category, level, message) in
        let regex = /threadsPerThreadgroup\s+(\(\d+,\s*\d+\))/
        if let match = message.firstMatch(of: regex) {
            let threadsPerThreadgroup = String(match.1)
            if !allThreadsPerThreadgroup.contains(threadsPerThreadgroup) {
                print(message)
                allThreadsPerThreadgroup.insert(threadsPerThreadgroup)
            }
        }
    }
    commandBufferOptions.logState = logState
    
    let commandAllocator = device.makeCommandAllocator()!
    commandBuffer.beginCommandBuffer(allocator: commandAllocator, options: commandBufferOptions)
    
    let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
    
    let library = try! device.makeDefaultLibrary(bundle: .main)
    let kernelFunction = library.makeFunction(name: "metalThreadgroup2D")!
    
    let computePipelineState = try! await device.makeComputePipelineState(function: kernelFunction)
    computeCommandEncoder.setComputePipelineState(computePipelineState)
    
    // (1) print max total threads per threadgroup and thread execution width
    print("max total threads per threadgroup: \(computePipelineState.maxTotalThreadsPerThreadgroup)")
    print("thread execution width: \(computePipelineState.threadExecutionWidth)")
    
    // (2) Calculate the threads per threadgroup based on max total threads per threadgroup and thread execution width
    let threadsPerGrid = MTLSizeMake(gridWidth, gridHeight, 1)
    let threadsPerThreadgroup = MTLSizeMake(computePipelineState.threadExecutionWidth, computePipelineState.maxTotalThreadsPerThreadgroup / computePipelineState.threadExecutionWidth, 1)
    
    computeCommandEncoder.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    
    computeCommandEncoder.endEncoding()
    
    commandBuffer.endCommandBuffer()
    
    let commandQueue = device.makeMTL4CommandQueue()!
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
}

private func metalThreadgroupUniformity(gridWidth: Int, gridHeight: Int) async {
    let device = MTLCreateSystemDefaultDevice()!
    
    let commandBuffer = device.makeCommandBuffer()!
    
    let commandBufferOptions = MTL4CommandBufferOptions()
    let logStateDescriptor = MTLLogStateDescriptor()
    logStateDescriptor.level = .info
    logStateDescriptor.bufferSize = 1024 * 1024 * 1024
    let logState = try! device.makeLogState(descriptor: logStateDescriptor)
    var allThreadsPerThreadgroup = Set<String>()
    logState.addLogHandler { (subsystem, category, level, message) in
        let regex = /threadsPerThreadgroup\s+(\(\d+,\s*\d+\))/
        if let match = message.firstMatch(of: regex) {
            let threadsPerThreadgroup = String(match.1)
            if !allThreadsPerThreadgroup.contains(threadsPerThreadgroup) {
                print(message)
                allThreadsPerThreadgroup.insert(threadsPerThreadgroup)
            }
        }
    }
    commandBufferOptions.logState = logState
    
    let commandAllocator = device.makeCommandAllocator()!
    commandBuffer.beginCommandBuffer(allocator: commandAllocator, options: commandBufferOptions)
    
    let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
    
    let library = try! device.makeDefaultLibrary(bundle: .main)
    let kernelFunction = library.makeFunction(name: "metalThreadgroupUniformity")!
    
    let computePipelineState = try! await device.makeComputePipelineState(function: kernelFunction)
    computeCommandEncoder.setComputePipelineState(computePipelineState)

    // (1) Calculate the threads per threadgroup based on max total threads per threadgroup and thread execution width
    let threadsPerThreadgroup = MTLSizeMake(computePipelineState.threadExecutionWidth, computePipelineState.maxTotalThreadsPerThreadgroup / computePipelineState.threadExecutionWidth, 1)
    let threadsPerGrid = MTLSizeMake(gridWidth, gridHeight, 1)
    
    // (2) encode a compute dispatch command with dispatchThreads
    computeCommandEncoder.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    
    computeCommandEncoder.endEncoding()
    
    commandBuffer.endCommandBuffer()
    
    let commandQueue = device.makeMTL4CommandQueue()!
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
}

private func metalSimdgroup(gridWidth: Int, gridHeight: Int) async {
    let device = MTLCreateSystemDefaultDevice()!
    
    let commandBuffer = device.makeCommandBuffer()!
    
    let commandBufferOptions = MTL4CommandBufferOptions()
    let logStateDescriptor = MTLLogStateDescriptor()
    logStateDescriptor.level = .info
    logStateDescriptor.bufferSize = 1024 * 1024 * 1024
    let logState = try! device.makeLogState(descriptor: logStateDescriptor)
    var allSimdgroupsPerThreadgroup = Set<String>()
    logState.addLogHandler { (subsystem, category, level, message) in
        let regex = /\s+simdgroupsPerThreadgroup\s+(\d+)/
        
        if let match = message.firstMatch(of: regex) {
            let simdgroupsPerThreadgroup = String(match.1)
            if !allSimdgroupsPerThreadgroup.contains(simdgroupsPerThreadgroup) {
                print(message)
                allSimdgroupsPerThreadgroup.insert(simdgroupsPerThreadgroup)
            }
        }
    }
    commandBufferOptions.logState = logState
    
    let commandAllocator = device.makeCommandAllocator()!
    commandBuffer.beginCommandBuffer(allocator: commandAllocator, options: commandBufferOptions)
    
    let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
    
    let library = try! device.makeDefaultLibrary(bundle: .main)
    let kernelFunction = library.makeFunction(name: "metalSimdgroup")!
    
    let computePipelineState = try! await device.makeComputePipelineState(function: kernelFunction)
    computeCommandEncoder.setComputePipelineState(computePipelineState)

    // (1) Calculate the threads per threadgroup based on max total threads per threadgroup and thread execution width
    let threadsPerThreadgroup = MTLSizeMake(computePipelineState.threadExecutionWidth, computePipelineState.maxTotalThreadsPerThreadgroup / computePipelineState.threadExecutionWidth, 1)
    let threadsPerGrid = MTLSizeMake(gridWidth, gridHeight, 1)
    
    // (2) encode a compute dispatch command with dispatchThreads
    computeCommandEncoder.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    
    computeCommandEncoder.endEncoding()
    
    commandBuffer.endCommandBuffer()
    
    let commandQueue = device.makeMTL4CommandQueue()!
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
}

func metalThreadDispatching() async {
    print("Metal Grid 1D")
    await metalGrid1D()
    
    print("\nMetal Grid 2D")
    await metalGrid2D()
    
    print("\nMetal Grid 3D")
    await metalGrid3D()
    
    print("\nMetal Threadgroup 1D")
    await metalThreadgroup1D()
    
    print("\nMetal Threadgroup 2D")
    await metalThreadgroup2D()
    
    print("\nMetal Threadgroup 3D")
    await metalThreadgroup3D()

    print("\nMetal Threadgroup Sizing 1D 2048")
    await metalThreadgroupSizing1D(gridWidth: 2048)
    print("\nMetal Threadgroup Sizing 1D 16")
    await metalThreadgroupSizing1D(gridWidth: 16)

    print("\nMetal Threadgroup Sizing 2D 64x64")
    await metalThreadgroupSizing2D(gridWidth: 64, gridHeight: 64)
    print("\nMetal Threadgroup Sizing 2D 16x8")
    await metalThreadgroupSizing2D(gridWidth: 16, gridHeight: 8)

    print("\nMetal Threadgroup Uniformity 32x64")
    await metalThreadgroupUniformity(gridWidth: 32, gridHeight: 64)
    print("\nMetal Threadgroup Uniformity 35x47")
    await metalThreadgroupUniformity(gridWidth: 35, gridHeight: 47)
    
    print("\nMetal SIMD-group Threadgroup 32x64")
    await metalSimdgroup(gridWidth: 32, gridHeight: 64)
    print("\nMetal SIMD-group Threadgroup 35x47")
    await metalSimdgroup(gridWidth: 35, gridHeight: 47)
}
