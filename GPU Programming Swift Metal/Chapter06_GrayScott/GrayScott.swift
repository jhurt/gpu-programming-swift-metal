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

import Combine
import MetalKit
import SwiftUI

struct GrayScottParams {
    var F: Float = 0.0 // feed rate
    var K: Float = 0.0 // kill rate
    var Du: Float = 1.0 // U diffusion coefficient
    var Dv: Float = 0.5 // V diffusion coefficient
    var dt: Float = 1.0 // time step
}

class GrayScott {
    let device: MTLDevice
    let commandQueue: MTL4CommandQueue
    let commandBuffer: MTL4CommandBuffer
    let commandAllocator: MTL4CommandAllocator
    
    var uBuffers: [MTLBuffer] = []
    var vBuffers: [MTLBuffer] = []
    var currentBufferIndex = 0
    var cellEnvironmentsBuffer: MTLBuffer
    
    let reactionDiffusionComputePipelineState: MTLComputePipelineState
    let reactionDiffusionArgumentTable: MTL4ArgumentTable
    
    let reactionDiffusionMultipleEnvironmentComputePipelineState: MTLComputePipelineState
    let reactionDiffusionMultipleEnvironmentArgumentTable: MTL4ArgumentTable
    
    let gridWidth = 128
    let gridHeight = 128
    var gridSize: SIMD2<UInt32>
    var environmentCount: UInt32
    
    var running: Bool = false
    var stepCount: Int = 0
    var averageTime: Double = 0.0
    
    init() async {
        device = MTLCreateSystemDefaultDevice()!
        
        // (1) initialize chemicals U and V
        for _ in 0..<2 {
            uBuffers.append(device.makeBuffer(length: gridWidth * gridHeight * MemoryLayout<Float>.stride, options: .storageModeShared)!)
            vBuffers.append(device.makeBuffer(length: gridWidth * gridHeight * MemoryLayout<Float>.stride, options: .storageModeShared)!)
        }
        let ptrU = uBuffers[0].contents().bindMemory(to: Float.self, capacity: gridWidth * gridHeight)
        let ptrV = vBuffers[0].contents().bindMemory(to: Float.self, capacity: gridWidth * gridHeight)
        for i in 0..<gridWidth * gridHeight {
            ptrU[i] = 1.0
            ptrV[i] = 0.0
        }
        
        // (2) seed the reaction in the middle of the grid
        let cx = gridWidth / 2
        let cy = gridHeight / 2
        let seedWidth = Int.random(in: 10...20)
        let seedHeight = Int.random(in: 10...20)
        for y in (cy - seedHeight)..<(cy + seedHeight) {
            for x in (cx - seedWidth)..<(cx + seedWidth) {
                let idx = y * gridWidth + x
                ptrV[idx] = Float.random(in: 0.5...1.0)
            }
        }
        
        // (3) copy U and V for double buffering
        let ptrU2 = uBuffers[1].contents()
        let ptrV2 = vBuffers[1].contents()
        ptrU2.copyMemory(from: uBuffers[0].contents(), byteCount: gridWidth * gridHeight * MemoryLayout<Float>.stride)
        ptrV2.copyMemory(from: vBuffers[0].contents(), byteCount: gridWidth * gridHeight * MemoryLayout<Float>.stride)
        
        gridSize = SIMD2<UInt32>(UInt32(gridWidth), UInt32(gridHeight))
        let gridSizeBuffer = device.makeBuffer(bytes: &gridSize, length: MemoryLayout<SIMD2<UInt32>>.stride, options: .storageModeShared)!
        
        // (4) choose the coral parameters for the V reactant
        var coralParams = GrayScottEnvironment.coral
        let coralParamsBuffer = device.makeBuffer(bytes: &coralParams, length: MemoryLayout<GrayScottParams>.stride, options: .storageModeShared)!
        
        // (1) get a handle to the `reactionDiffusion` kernel function
        let library = device.makeDefaultLibrary()!
        let reactionDiffusionFunction = library.makeFunction(name: "reactionDiffusion")!
        reactionDiffusionComputePipelineState = try! await device.makeComputePipelineState(function: reactionDiffusionFunction)
        
        // (2) create the argument table for the `reactionDiffusion` kernel function
        let reactionDiffusionArgumentTableDescriptor = MTL4ArgumentTableDescriptor()
        reactionDiffusionArgumentTableDescriptor.maxBufferBindCount = 6
        reactionDiffusionArgumentTable = try! device.makeArgumentTable(descriptor: reactionDiffusionArgumentTableDescriptor)
        reactionDiffusionArgumentTable.setAddress(uBuffers[0].gpuAddress, index: 0)
        reactionDiffusionArgumentTable.setAddress(vBuffers[0].gpuAddress, index: 1)
        reactionDiffusionArgumentTable.setAddress(uBuffers[1].gpuAddress, index: 2)
        reactionDiffusionArgumentTable.setAddress(vBuffers[1].gpuAddress, index: 3)
        reactionDiffusionArgumentTable.setAddress(gridSizeBuffer.gpuAddress, index: 4)
        reactionDiffusionArgumentTable.setAddress(coralParamsBuffer.gpuAddress, index: 5)
        
        // (4) create an array of multiple environments
        var environments: [GrayScottParams] = [
            GrayScottEnvironment.coral,
            GrayScottEnvironment.mitosis,
            GrayScottEnvironment.fingerprints,
            GrayScottEnvironment.solitons,
            GrayScottEnvironment.chaos,
        ]
        environments.shuffle()
        let environmentsBuffer = device.makeBuffer(bytes: &environments, length: MemoryLayout<GrayScottParams>.stride * environments.count, options: .storageModeShared)!
        environmentCount = UInt32(environments.count)
        let environmentCountBuffer = device.makeBuffer(bytes: &environmentCount, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        
        // (5) assign each cell in the grid to a environments
        var cellEnvironments = GrayScottEnvironment.getEnvironmentsForGrid(totalEnvironments: environments.count, gridWidth: gridWidth, gridHeight: gridHeight)
        cellEnvironmentsBuffer = device.makeBuffer(bytes: &cellEnvironments, length: MemoryLayout<UInt32>.stride * cellEnvironments.count, options: .storageModeShared)!
        
        // (6) get handles to the visible functions the kernel will read from a function table
        let isotropicDiffusionFunction = library.makeFunction(name: "isotropicDiffusion")!
        let anisotropicDiffusionFunction = library.makeFunction(name: "anisotropicDiffusion")!
        
        // (7) get a handle to the `reactionDiffusionMultipleEnvironment` kernel function
        let reactionDiffusionMultipleEnvironmentFunction = library.makeFunction(name: "reactionDiffusionMultipleEnvironment")!
        
        // (8) create a compute pipeline state for the diffusion kernel, telling it the functions we want it to link to
        let reactionDiffusionMultipleEnvironmentComputePipelineDescriptor = MTLComputePipelineDescriptor()
        reactionDiffusionMultipleEnvironmentComputePipelineDescriptor.computeFunction = reactionDiffusionMultipleEnvironmentFunction
        let linkedFunctions = MTLLinkedFunctions()
        linkedFunctions.functions = [isotropicDiffusionFunction, anisotropicDiffusionFunction]
        reactionDiffusionMultipleEnvironmentComputePipelineDescriptor.linkedFunctions = linkedFunctions
        reactionDiffusionMultipleEnvironmentComputePipelineState = try! device.makeComputePipelineState(descriptor: reactionDiffusionMultipleEnvironmentComputePipelineDescriptor, options: [], reflection: nil)
        
        // (9) create a visible function table that will be passed to the diffusion kernel
        let diffusionFunctionTableDesc = MTLVisibleFunctionTableDescriptor()
        diffusionFunctionTableDesc.functionCount = 2
        let diffusionFunctionTable = reactionDiffusionMultipleEnvironmentComputePipelineState.makeVisibleFunctionTable(descriptor: diffusionFunctionTableDesc)!
        let isotropicDiffusionFunctionHandle = reactionDiffusionMultipleEnvironmentComputePipelineState.functionHandle(function: isotropicDiffusionFunction)
        let anisotropicDiffusionFunctionHandle = reactionDiffusionMultipleEnvironmentComputePipelineState.functionHandle(function: anisotropicDiffusionFunction)
        diffusionFunctionTable.setFunction(isotropicDiffusionFunctionHandle, index: 0)
        diffusionFunctionTable.setFunction(anisotropicDiffusionFunctionHandle, index: 1)
        
        // (10) create the argument table for the `reactionDiffusionMultipleEnvironment` kernel function
        let reactionDiffusionMultipleEnvironmentArgumentTableDescriptor = MTL4ArgumentTableDescriptor()
        reactionDiffusionMultipleEnvironmentArgumentTableDescriptor.maxBufferBindCount = 9
        reactionDiffusionMultipleEnvironmentArgumentTable = try! device.makeArgumentTable(descriptor: reactionDiffusionMultipleEnvironmentArgumentTableDescriptor)
        reactionDiffusionMultipleEnvironmentArgumentTable.setAddress(uBuffers[0].gpuAddress, index: 0)
        reactionDiffusionMultipleEnvironmentArgumentTable.setAddress(vBuffers[0].gpuAddress, index: 1)
        reactionDiffusionMultipleEnvironmentArgumentTable.setAddress(uBuffers[1].gpuAddress, index: 2)
        reactionDiffusionMultipleEnvironmentArgumentTable.setAddress(vBuffers[1].gpuAddress, index: 3)
        reactionDiffusionMultipleEnvironmentArgumentTable.setAddress(gridSizeBuffer.gpuAddress, index: 4)
        reactionDiffusionMultipleEnvironmentArgumentTable.setAddress(environmentsBuffer.gpuAddress, index: 5)
        reactionDiffusionMultipleEnvironmentArgumentTable.setAddress(environmentCountBuffer.gpuAddress, index: 6)
        reactionDiffusionMultipleEnvironmentArgumentTable.setAddress(cellEnvironmentsBuffer.gpuAddress, index: 7)
        // (11) bind the function table to the argument table
        reactionDiffusionMultipleEnvironmentArgumentTable.setResource(diffusionFunctionTable.gpuResourceID, bufferIndex: 8)
        
        let residencySet = try! device.makeResidencySet(descriptor: .init())
        residencySet.addAllocation(uBuffers[0])
        residencySet.addAllocation(vBuffers[0])
        residencySet.addAllocation(uBuffers[1])
        residencySet.addAllocation(vBuffers[1])
        residencySet.addAllocation(gridSizeBuffer)
        residencySet.addAllocation(coralParamsBuffer)
        residencySet.addAllocation(environmentsBuffer)
        residencySet.addAllocation(environmentCountBuffer)
        residencySet.addAllocation(cellEnvironmentsBuffer)
        // (12) add the function table to the residency set. this is not strictly necessary but increases performance
        residencySet.addAllocation(diffusionFunctionTable)
        residencySet.commit()
        
        commandQueue = device.makeMTL4CommandQueue()!
        commandQueue.addResidencySet(residencySet)
        commandBuffer = device.makeCommandBuffer()!
        commandAllocator = device.makeCommandAllocator()!
    }
    
    private func singleEnvironment() {
        // (1) re-use a single command buffer and allocator
        commandBuffer.beginCommandBuffer(allocator: commandAllocator, options: .init())
        let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
        // (2) re-use the compute pipeline state and argument table
        computeCommandEncoder.setComputePipelineState(reactionDiffusionComputePipelineState)
        computeCommandEncoder.setArgumentTable(reactionDiffusionArgumentTable)
        
        // (3) dispatch on as many threads per threadgroup as Metal allows
        let threadsPerThreadgroupWidth = reactionDiffusionComputePipelineState.threadExecutionWidth
        let threadsPerThreadgroupHeight = reactionDiffusionComputePipelineState.maxTotalThreadsPerThreadgroup / threadsPerThreadgroupWidth
        let threadsPerThreadgroup = MTLSizeMake(threadsPerThreadgroupWidth, threadsPerThreadgroupHeight, 1)
        // (4) threadgroup count is based on simulation grid size and threads per threadgroup
        let threadgroups = MTLSizeMake((gridWidth + threadsPerThreadgroupWidth - 1) / threadsPerThreadgroupWidth,
                                       (gridHeight + threadsPerThreadgroupHeight - 1) / threadsPerThreadgroupHeight,
                                       1)
        computeCommandEncoder.dispatchThreadgroups(threadgroupsPerGrid: threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        computeCommandEncoder.endEncoding()
        commandBuffer.endCommandBuffer()
        
        // (5) submit the dispatch command and block on the completion
        let semaphore = DispatchSemaphore(value: 0)
        let commitOptions = MTL4CommitOptions()
        commitOptions.addFeedbackHandler { feedback in
            if let error = feedback.error {
                fatalError(error.localizedDescription)
            }
            semaphore.signal()
        }
        commandQueue.commit([commandBuffer], options: commitOptions)
        semaphore.wait()
        
        // (6) swap the current and next states of U and V so
        // that current state points to what the kernel just wrote to next state
        let nextIndex = (currentBufferIndex + 1) % 2
        if nextIndex == 0 {
            reactionDiffusionArgumentTable.setAddress(uBuffers[0].gpuAddress, index: 0)
            reactionDiffusionArgumentTable.setAddress(vBuffers[0].gpuAddress, index: 1)
            reactionDiffusionArgumentTable.setAddress(uBuffers[1].gpuAddress, index: 2)
            reactionDiffusionArgumentTable.setAddress(vBuffers[1].gpuAddress, index: 3)
        } else {
            reactionDiffusionArgumentTable.setAddress(uBuffers[1].gpuAddress, index: 0)
            reactionDiffusionArgumentTable.setAddress(vBuffers[1].gpuAddress, index: 1)
            reactionDiffusionArgumentTable.setAddress(uBuffers[0].gpuAddress, index: 2)
            reactionDiffusionArgumentTable.setAddress(vBuffers[0].gpuAddress, index: 3)
        }
        currentBufferIndex = nextIndex
    }
    
    private func multipleEnvironment() {
        commandBuffer.beginCommandBuffer(allocator: commandAllocator, options: .init())
        let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
        computeCommandEncoder.setComputePipelineState(reactionDiffusionMultipleEnvironmentComputePipelineState)
        computeCommandEncoder.setArgumentTable(reactionDiffusionMultipleEnvironmentArgumentTable)
        
        let threadsPerThreadgroupWidth = reactionDiffusionComputePipelineState.threadExecutionWidth
        let threadsPerThreadgroupHeight = reactionDiffusionComputePipelineState.maxTotalThreadsPerThreadgroup / threadsPerThreadgroupWidth
        let threadsPerThreadgroup = MTLSizeMake(threadsPerThreadgroupWidth, threadsPerThreadgroupHeight, 1)
        let threadgroups = MTLSizeMake((gridWidth + threadsPerThreadgroupWidth - 1) / threadsPerThreadgroupWidth,
                                       (gridHeight + threadsPerThreadgroupHeight - 1) / threadsPerThreadgroupHeight,
                                       1)
        
        computeCommandEncoder.dispatchThreadgroups(threadgroupsPerGrid: threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        computeCommandEncoder.endEncoding()
        commandBuffer.endCommandBuffer()
        
        let semaphore = DispatchSemaphore(value: 0)
        let commitOptions = MTL4CommitOptions()
        commitOptions.addFeedbackHandler { feedback in
            if let error = feedback.error {
                fatalError(error.localizedDescription)
            }
            semaphore.signal()
        }
        commandQueue.commit([commandBuffer], options: commitOptions)
        semaphore.wait()
        
        let nextIndex = (currentBufferIndex + 1) % 2
        if nextIndex == 0 {
            reactionDiffusionMultipleEnvironmentArgumentTable.setAddress(uBuffers[0].gpuAddress, index: 0)
            reactionDiffusionMultipleEnvironmentArgumentTable.setAddress(vBuffers[0].gpuAddress, index: 1)
            reactionDiffusionMultipleEnvironmentArgumentTable.setAddress(uBuffers[1].gpuAddress, index: 2)
            reactionDiffusionMultipleEnvironmentArgumentTable.setAddress(vBuffers[1].gpuAddress, index: 3)
        } else {
            reactionDiffusionMultipleEnvironmentArgumentTable.setAddress(uBuffers[1].gpuAddress, index: 0)
            reactionDiffusionMultipleEnvironmentArgumentTable.setAddress(vBuffers[1].gpuAddress, index: 1)
            reactionDiffusionMultipleEnvironmentArgumentTable.setAddress(uBuffers[0].gpuAddress, index: 2)
            reactionDiffusionMultipleEnvironmentArgumentTable.setAddress(vBuffers[0].gpuAddress, index: 3)
        }
        currentBufferIndex = nextIndex
    }
    
    func start() {
        if running {
            return
        }
        running = true
        stepCount = 0
        averageTime = 0.0
        currentBufferIndex = 0
        step()
    }
    
    func stop() {
        running = false
    }
    
    func step() {
        if !running {
            return
        }
        let startTime = CFAbsoluteTimeGetCurrent()
        //singleEnvironment()
        multipleEnvironment()
        let time = CFAbsoluteTimeGetCurrent() - startTime
        stepCount += 1
        averageTime = averageTime * (Double(stepCount - 1) / Double(stepCount)) + (time / Double(stepCount))
        if stepCount % 100 == 0 {
            print("Gray-Scott step \(stepCount) average time to run step: \(averageTime)s")
        }
    }
}
