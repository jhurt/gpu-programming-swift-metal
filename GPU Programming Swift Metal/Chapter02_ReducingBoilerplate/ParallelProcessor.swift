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

import Foundation
import Metal

struct ProcessOptions {
    var threadsPerGrid: Int = 1024
    var threadsPerGridMTLSize: MTLSize? = nil
    
    var threadgroupsPerGrid: Int?
    var threadgroupsPerGridMTLSize: MTLSize? = nil
    
    var threadsPerThreadgroup: Int?
    var threadsPerThreadgroupMTLSize: MTLSize? = nil
    
    var removeResidencySet: Bool = true
    var resetCommandAllocator: Bool = true
    
    var logBufferKB: Int = 0
}

class ParallelProcessor {
    var log: Bool = false
    var device: MTLDevice!
    var commandQueue: MTL4CommandQueue!
    var commandAllocator: MTL4CommandAllocator!
    var commandBuffer: MTL4CommandBuffer?
    var residencySet: MTLResidencySet?
    var library: MTLLibrary!
    var kernelNameToFunction: [String: MTLFunction] = [:]
    var kernelNameToPipelineState: [String: MTLComputePipelineState] = [:]
    
    convenience init() {
        self.init(log: false)
    }

    init(log: Bool) {
        device = MTLCreateSystemDefaultDevice()!
        self.log = log
        if log {
            print("ParallelProcessor Metal Device: \(device.name)")
        }

        commandQueue = device.makeMTL4CommandQueue()!
        commandAllocator = device.makeCommandAllocator()!

        library = try! device.makeDefaultLibrary(bundle: .main)
    }

    public func registerKernel(_ name: String, functionConstantValues: MTLFunctionConstantValues? = nil) async -> MTLComputePipelineState {
        // check the cache for a compute pipeline state for the kernel function
        if let kernelFunction = kernelNameToFunction[name] {
            if kernelFunction.name == name {
                return kernelNameToPipelineState[name]!
            }
        }
        
        // assign function constants if provided
        if let functionConstantValues = functionConstantValues {
            kernelNameToFunction[name] = try! await library.makeFunction(name: name, constantValues: functionConstantValues)
        } else {
            kernelNameToFunction[name] = library.makeFunction(name: name)!
        }
        
        // get a GPU pipeline configuration for running kernels in a compute pass
        let (computePipelineState, computePipelineReflection) = try! await device.makeComputePipelineState(function: kernelNameToFunction[name]!, options: [.bindingInfo, .bufferTypeInfo])
        if log {
            print("ParallelProcessor kernel name \(name), thread execution width \(computePipelineState.threadExecutionWidth), max total threads per threadgroup \(computePipelineState.maxTotalThreadsPerThreadgroup), static threadgroup memory length \(computePipelineState.staticThreadgroupMemoryLength)")
        }
        
        // cache the configuration
        kernelNameToPipelineState[name] = computePipelineState
        
        return computePipelineState
    }
    
    public func process(kernelName: String, buffers: MTLBuffer..., textures: MTLTexture..., threadgroupMemory: Int..., functionConstantValues: MTLFunctionConstantValues? = nil, options: ProcessOptions) async {
        // (1) lazy register a MTLComputePipelineState for the kernel
        let computePipelineState = await registerKernel(kernelName, functionConstantValues: functionConstantValues)
        // (2) create an argument table
        let argumentTableDescriptor = MTL4ArgumentTableDescriptor()
        argumentTableDescriptor.maxBufferBindCount = buffers.count
        argumentTableDescriptor.maxTextureBindCount = textures.count
        let argumentTable = try! device.makeArgumentTable(descriptor: argumentTableDescriptor)

        // (3) bind any buffers to the argument table
        for (index, buffer) in buffers.enumerated() {
            argumentTable.setAddress(buffer.gpuAddress, index: index)
        }
        
        // (4) bind any textures to the argument table
        for (index, texture) in textures.enumerated() {
            argumentTable.setTexture(texture.gpuResourceID, index: index)
        }
        
        // (5) lazy create a residency set
        if residencySet == nil {
            residencySet = try! device.makeResidencySet(descriptor: .init())
            commandQueue.addResidencySet(residencySet!)
        }
        
        // (6) allocate any buffers to the residency set
        for buffer in buffers {
            residencySet!.addAllocation(buffer)
        }
        
        // (7) allocate any textures to the residency set
        for texture in textures {
            residencySet!.addAllocation(texture)
        }
        residencySet!.commit()
        
        // (8) add log handler if `logBufferKB` is set
        let commandBufferOptions = MTL4CommandBufferOptions()
        if options.logBufferKB != 0 {
            let logStateDescriptor = MTLLogStateDescriptor()
            logStateDescriptor.level = .info
            logStateDescriptor.bufferSize = options.logBufferKB * 1024
            let logState = try! device.makeLogState(descriptor: logStateDescriptor)
            logState.addLogHandler { (subsystem, category, level, message) in
                print("[GPU] \(subsystem ?? "default")/\(category ?? "default"): \(message)")
            }
            commandBufferOptions.logState = logState
        }
        
        // (9) lazy create a command buffer
        if commandBuffer == nil {
            commandBuffer = device.makeCommandBuffer()!
        }
        
        // (10) attach the command buffer to the command allocator
        commandBuffer!.beginCommandBuffer(allocator: commandAllocator, options: commandBufferOptions)
        commandBuffer!.label = kernelName
        
        // (11) get a new command encoder
        let computeCommandEncoder = commandBuffer!.makeComputeCommandEncoder()!
        computeCommandEncoder.label = kernelName
        computeCommandEncoder.setComputePipelineState(computePipelineState)
        computeCommandEncoder.setArgumentTable(argumentTable)
        
        // (12) configure any threadgroup memory buffers
        for (index, threadgroupMemoryLength) in threadgroupMemory.enumerated() {
            computeCommandEncoder.setThreadgroupMemoryLength(threadgroupMemoryLength, index: index)
        }
        
        // (13) determine threads per threadgroup
        var threadsPerThreadgroupMTLSize: MTLSize
        if let tpt = options.threadsPerThreadgroupMTLSize {
            threadsPerThreadgroupMTLSize = tpt
        } else {
            if let threadsPerThreadgroup = options.threadsPerThreadgroup {
                threadsPerThreadgroupMTLSize = MTLSizeMake(threadsPerThreadgroup, 1, 1)
            } else {
                threadsPerThreadgroupMTLSize = MTLSizeMake(computePipelineState.threadExecutionWidth, 1, 1)
            }
        }
        
        // (14) encode a compute dispatch command
        var threadgroupsPerGridMTLSize: MTLSize? = nil
        if let threadgroupsPerGrid = options.threadgroupsPerGrid {
            threadgroupsPerGridMTLSize = MTLSizeMake(threadgroupsPerGrid, 1, 1)
        } else {
            threadgroupsPerGridMTLSize = options.threadgroupsPerGridMTLSize
        }
        if let threadgroupsPerGridMTLSize = threadgroupsPerGridMTLSize {
            if log {
                print("ParallelProcessor: encoding dispatch threadgroups, threadgroupsPerGrid: \(threadgroupsPerGridMTLSize), threadsPerThreadgroup \(threadsPerThreadgroupMTLSize)")
            }
            computeCommandEncoder.dispatchThreadgroups(threadgroupsPerGrid: threadgroupsPerGridMTLSize, threadsPerThreadgroup: threadsPerThreadgroupMTLSize)
        } else {
            var threadsPerGridMTLSize: MTLSize
            if let tpg = options.threadsPerGridMTLSize {
                threadsPerGridMTLSize = tpg
            } else {
                threadsPerGridMTLSize = MTLSizeMake(options.threadsPerGrid, 1, 1)
            }
            if log {
                print("ParallelProcessor: encoding dispatch threads, threadsPerGrid: \(threadsPerGridMTLSize), threadsPerThreadgroup \(threadsPerThreadgroupMTLSize)")
            }
            computeCommandEncoder.dispatchThreads(threadsPerGrid: threadsPerGridMTLSize, threadsPerThreadgroup: threadsPerThreadgroupMTLSize)
        }
        computeCommandEncoder.endEncoding()
        commandBuffer!.endCommandBuffer()
        
        let commitOptions = MTL4CommitOptions()
        try! await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            commitOptions.addFeedbackHandler { [weak self] feedback in
                if let error = feedback.error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume()
                }
                
                // (15) log GPU time for any committed command buffers
                guard let self = self else {
                    return
                }
                if log {
                    print("ParallelProcessor \(kernelName) GPU time: \(feedback.gpuEndTime - feedback.gpuStartTime) seconds")
                }
            }
            
            // (16) enqueue the compute command encoder with the compute pass
            commandQueue.commit([commandBuffer!], options: commitOptions)
        }
        
        // (17) possibly cleanup the residency set
        if options.removeResidencySet {
            if let residencySet = residencySet {
                commandQueue.removeResidencySet(residencySet)
                self.residencySet = nil
            }
        }
        
        // (18) possibly reset the command allocator
        if options.resetCommandAllocator {
            commandAllocator.reset()
        }
    }
}
