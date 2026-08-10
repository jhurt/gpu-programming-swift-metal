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

func forLoopProfiling() async {
    let device = MTLCreateSystemDefaultDevice()!
    
    let captureManager = MTLCaptureManager.shared()
    let descriptor = MTLCaptureDescriptor()
    descriptor.captureObject = device
    descriptor.destination = .developerTools
    try! captureManager.startCapture(with: descriptor)
    
    let inputLength = 4_000_000
    var input = await ParallelData.randomFloatArray(length: inputLength)
    let buffer = device.makeBuffer(bytesNoCopy: &input, length: MemoryLayout<Float>.stride * inputLength, options: .storageModeShared)!
    
    let argumentTableDescriptor = MTL4ArgumentTableDescriptor()
    argumentTableDescriptor.maxBufferBindCount = 1
    let argumentTable = try! device.makeArgumentTable(descriptor: argumentTableDescriptor)
    argumentTable.setAddress(buffer.gpuAddress, index: 0)
    
    
    let library = try! device.makeDefaultLibrary(bundle: .main)
    
    let residencySet = try! device.makeResidencySet(descriptor: .init())
    residencySet.addAllocation(buffer)
    residencySet.commit()
    
    // run twice to ignore the first run GPU timings of each kernel/function constant pair
    for _ in 0..<2 {
        var iterations = [1, 8, 32, 2048, 32768]
        for functionName in ["regularForLoop", "divergentForLoop", "unrolledForLoop"] {
            for i in 0..<iterations.count  {
                let commandBuffer = device.makeCommandBuffer()!
                let commandAllocator = device.makeCommandAllocator()!
                commandBuffer.beginCommandBuffer(allocator: commandAllocator)
                
                let functionConstantValues = MTLFunctionConstantValues()
                functionConstantValues.setConstantValue(&iterations[i], type: .int, index: 0)
                let kernelFunction = try! await library.makeFunction(name: functionName, constantValues: functionConstantValues)
                
                let computePipelineDescriptor = MTLComputePipelineDescriptor()
                computePipelineDescriptor.computeFunction = kernelFunction
                computePipelineDescriptor.label = "\(functionName) \(iterations[i])"
                
                let computePipelineState = try! device.makeComputePipelineState(descriptor: computePipelineDescriptor, options: [], reflection: nil)
                
                let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
                computeCommandEncoder.setComputePipelineState(computePipelineState)
                computeCommandEncoder.setArgumentTable(argumentTable)
                computeCommandEncoder.dispatchThreads(threadsPerGrid: MTLSizeMake(inputLength, 1, 1),
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
                        
                        let gpuTime = feedback.gpuEndTime - feedback.gpuStartTime
                        print("\(functionName) \(iterations[i]) GPU Time: \(String(format: "%.6f", gpuTime)) seconds")
                    }
                    commandQueue.commit([commandBuffer], options: commitOptions)
                }
            }
        }
    }
    
    captureManager.stopCapture()
}
