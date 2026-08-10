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

private func memoryBandwidth(inputLength: Int) async {
    let device = MTLCreateSystemDefaultDevice()!
    
    let counterSets = device.counterSets!
    for counterSet in counterSets {
        print("GPU device \"\(device.name)\" supports the \"\(counterSet.name)\" counter set.")
        for counter in counterSet.counters {
            print("Counter set \"\(counterSet.name)\" contains the \"\(counter.name)\" counter.")
        }
    }

    // (1) get a handle to the capture manager
    let captureManager = MTLCaptureManager.shared()

    // (2) create a capture descriptor
    let descriptor = MTLCaptureDescriptor()
    
    // (3) bind the Metal device to the descriptor
    descriptor.captureObject = device
    
    // (4) specify the destination as either Xcode developer tools or a GPU trace document
    descriptor.destination = .developerTools
    
    // (5) start the capture
    try! captureManager.startCapture(with: descriptor)
    
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
    
    let totalBytes = inputBuffer.length * dispatchCount * 2
    try! await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
        commitOptions.addFeedbackHandler { feedback in
            if let error = feedback.error {
                continuation.resume(throwing: error)
            } else {
                continuation.resume()
            }
            
            let gpuTime = feedback.gpuEndTime - feedback.gpuStartTime
            print("GPU Time: \(String(format: "%.4f", gpuTime)) seconds")
            
            let gigabytes = Double(totalBytes) / (1024.0 * 1024.0 * 1024.0)
            let bandwidth = gigabytes / gpuTime
            print("Throughput: \(String(format: "%.2f", bandwidth)) GB/s")
        }
        commandQueue.commit([commandBuffer], options: commitOptions)
    }
    
    // (6) stop the capture
    captureManager.stopCapture()
}

func metalProfiling() async {
    print("\nMemory Bandwidth Profiling")
    let bandwidthInputLength = 4_000_000
    await memoryBandwidth(inputLength: bandwidthInputLength)
}
