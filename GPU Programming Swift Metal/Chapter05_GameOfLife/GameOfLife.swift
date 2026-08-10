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

class GameOfLife {
    let device: MTLDevice

    var interestingTileList: [UInt32]
    let interestingTileListBuffer: MTLBuffer

    var interestingTileCount: UInt32
    let interestingTileCountBuffer: MTLBuffer

    var currentGridBuffer: MTLBuffer

    var nextGridBuffer: MTLBuffer
    var gridSizePixels: SIMD2<UInt32>
    let gridSizePixelsBuffer: MTLBuffer

    var gridSizeTiles: SIMD2<UInt32>
    let gridSizeTilesBuffer: MTLBuffer

    let commandQueue: MTL4CommandQueue
    let commandBuffer: MTL4CommandBuffer
    let commandAllocator: MTL4CommandAllocator
    
    let classifyTilesComputePipelineState: MTLComputePipelineState
    let classifyTilesArgumentTable: MTL4ArgumentTable
    let executeSimulationComputePipelineState: MTLComputePipelineState
    let executeSimulationArgumentTable: MTL4ArgumentTable
    
    let classifyTilesICBComputePipelineState: MTLComputePipelineState
    let classifyTilesICBArgumentTable: MTL4ArgumentTable
    let executeSimulationICBComputePipelineState: MTLComputePipelineState
    let indirectCommandBuffer: MTLIndirectCommandBuffer
    
    let gridWidth = 1024
    let gridHeight = 1024
    let tileWidth = 16
    let tileHeight = 16
    let tileCountX: Int
    let tileCountY: Int
    
    var running: Bool = false
    var stepCount: Int = 0
    var averageTime: Double = 0.0
    
    var completedClassifyThreadgroupCount: UInt32 = 0
    var completedClassifyThreadgroupCountBuffer: MTLBuffer
    
    init() async {
        // (1) calculate the number of tiles on the X and Y axis
        tileCountX = gridWidth / tileWidth
        tileCountY = gridHeight / tileHeight
        
        device = MTLCreateSystemDefaultDevice()!
        let library = try! device.makeDefaultLibrary(bundle: .main)

        // (2) allocate buffers for tracking the interesting tiles at each time step
        interestingTileList = [UInt32](repeating: 0, count: Int(tileCountX * tileCountY))
        interestingTileListBuffer = device.makeBuffer(bytesNoCopy: &interestingTileList, length: MemoryLayout<UInt32>.stride * tileCountX * tileCountY, options: .storageModeShared)!
        interestingTileCount = 0
        interestingTileCountBuffer = device.makeBuffer(bytesNoCopy: &interestingTileCount, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        
        // (3) allocate two grids, one for reading state at time step t and one for writing state at time step t + 1
        var currentGrid = GameOfLifeStartStates.createGosperGliderGunGrid(width: gridWidth, height: gridHeight)
        currentGridBuffer = device.makeBuffer(bytes: &currentGrid, length: MemoryLayout<UInt8>.stride * gridWidth * gridHeight, options: .storageModeShared)!
        var nextGrid = [UInt8](repeating: 0, count: gridWidth * gridHeight)
        nextGridBuffer = device.makeBuffer(bytes: &nextGrid, length: MemoryLayout<UInt8>.stride * gridWidth * gridHeight, options: .storageModeShared)!
        
        // (4) allocate buffers for the number of cells per grid and number of tiles per grid
        gridSizePixels = SIMD2<UInt32>(UInt32(gridWidth), UInt32(gridHeight))
        gridSizePixelsBuffer = device.makeBuffer(bytes: &gridSizePixels, length: MemoryLayout<SIMD2<UInt32>>.stride, options: .storageModeShared)!
        gridSizeTiles = SIMD2<UInt32>(UInt32(tileCountX), UInt32(tileCountY))
        gridSizeTilesBuffer = device.makeBuffer(bytes: &gridSizeTiles, length: MemoryLayout<SIMD2<UInt32>>.stride, options: .storageModeShared)!
        
        // (5) use additional atomic int to sync threadgroups so that only one thread encodes the indirect dispatch command for the simulation kernel
        completedClassifyThreadgroupCount = 0
        completedClassifyThreadgroupCountBuffer = device.makeBuffer(bytesNoCopy: &completedClassifyThreadgroupCount, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        
        // (1) get a handle to the classification kernel
        let classifyTilesFunction = library.makeFunction(name: "classifyTiles")!
        classifyTilesComputePipelineState = try! await device.makeComputePipelineState(function: classifyTilesFunction)

        // (2) create an argument table for the classification kernel
        let classifyTilesArgumentTableDescriptor = MTL4ArgumentTableDescriptor()
        classifyTilesArgumentTableDescriptor.maxBufferBindCount = 5
        classifyTilesArgumentTable = try! device.makeArgumentTable(descriptor: classifyTilesArgumentTableDescriptor)
        classifyTilesArgumentTable.setAddress(interestingTileListBuffer.gpuAddress, index: 0)
        classifyTilesArgumentTable.setAddress(interestingTileCountBuffer.gpuAddress, index: 1)
        classifyTilesArgumentTable.setAddress(currentGridBuffer.gpuAddress, index: 2)
        classifyTilesArgumentTable.setAddress(gridSizePixelsBuffer.gpuAddress, index: 3)
        classifyTilesArgumentTable.setAddress(gridSizeTilesBuffer.gpuAddress, index: 4)
        
        // (1) create an indirect command buffer that will encode dispatch threads for the `executeSimulation` kernel
        let indirectCommandBufferDescriptor = MTLIndirectCommandBufferDescriptor()
        indirectCommandBufferDescriptor.commandTypes = [.concurrentDispatch, .concurrentDispatchThreads]
        indirectCommandBufferDescriptor.inheritBuffers = false
        indirectCommandBufferDescriptor.inheritPipelineState = false
        indirectCommandBufferDescriptor.maxKernelBufferBindCount = 5
        let maxCommandCount = 1_000_000
        indirectCommandBuffer = device.makeIndirectCommandBuffer(
            descriptor: indirectCommandBufferDescriptor,
            maxCommandCount: maxCommandCount,
            options: .storageModeShared
        )!

        // (2) create a library function descriptor for the `executeSimulation` kernel function
        let executeSimulationICBKernelFunctionDescriptor = MTL4LibraryFunctionDescriptor()
        executeSimulationICBKernelFunctionDescriptor.library = library
        executeSimulationICBKernelFunctionDescriptor.name = "executeSimulation"

        // (3) create a compute pipeline descriptor for executing the `executeSimulation` kernel function from an indirect command buffer
        let executeSimulationICBComputePipelineDescriptor = MTL4ComputePipelineDescriptor()
        executeSimulationICBComputePipelineDescriptor.computeFunctionDescriptor = executeSimulationICBKernelFunctionDescriptor
        executeSimulationICBComputePipelineDescriptor.supportIndirectCommandBuffers = .enabled

        // (4) create a pipeline state for the `executeSimulation` kernel function whose dispatch commands can be encoded on an indirect command buffer.
        let metalCompiler = try! device.makeCompiler(descriptor: MTL4CompilerDescriptor())
        executeSimulationICBComputePipelineState = try! await metalCompiler.makeComputePipelineState(descriptor: executeSimulationICBComputePipelineDescriptor)
        
        // (5) create a pipeline state for the `classifyTilesICB` kernel
        let classifyTilesICBFunction = library.makeFunction(name: "classifyTilesICB")!
        classifyTilesICBComputePipelineState = try! await device.makeComputePipelineState(function: classifyTilesICBFunction)
        
        // (6) create an argument buffer that matches the shape of the `ClassifyTilesICBArguments` MSL struct
        let argumentEncoder = classifyTilesICBFunction.makeArgumentEncoder(bufferIndex: 0)
        let argumentBuffer = device.makeBuffer(length: argumentEncoder.encodedLength, options: .storageModeShared)!
        argumentEncoder.setArgumentBuffer(argumentBuffer, offset: 0)
        argumentEncoder.setIndirectCommandBuffer(indirectCommandBuffer, index: 0)
        argumentEncoder.setComputePipelineState(executeSimulationICBComputePipelineState, index: 1)
        
        // (7) create an argument table descriptor for the `classifyTilesICB` kernel function
        let classifyTilesICBArgumentTableDescriptor = MTL4ArgumentTableDescriptor()
        classifyTilesICBArgumentTableDescriptor.maxBufferBindCount = 8
        classifyTilesICBArgumentTable = try! device.makeArgumentTable(descriptor: classifyTilesICBArgumentTableDescriptor)
        classifyTilesICBArgumentTable.setAddress(argumentBuffer.gpuAddress, index: 0)
        classifyTilesICBArgumentTable.setAddress(interestingTileListBuffer.gpuAddress, index: 1)
        classifyTilesICBArgumentTable.setAddress(interestingTileCountBuffer.gpuAddress, index: 2)
        classifyTilesICBArgumentTable.setAddress(currentGridBuffer.gpuAddress, index: 3)
        classifyTilesICBArgumentTable.setAddress(nextGridBuffer.gpuAddress, index: 4)
        classifyTilesICBArgumentTable.setAddress(gridSizePixelsBuffer.gpuAddress, index: 5)
        classifyTilesICBArgumentTable.setAddress(gridSizeTilesBuffer.gpuAddress, index: 6)
        classifyTilesICBArgumentTable.setAddress(completedClassifyThreadgroupCountBuffer.gpuAddress, index: 7)
        
        // (3) get a handle to the simulation kernel
        let executeSimulationFunction = library.makeFunction(name: "executeSimulation")!
        executeSimulationComputePipelineState = try! await device.makeComputePipelineState(function: executeSimulationFunction)

        // (4) create an argument table for the simulation kernel
        let executeSimulationArgumentTableDescriptor = MTL4ArgumentTableDescriptor()
        executeSimulationArgumentTableDescriptor.maxBufferBindCount = 5
        executeSimulationArgumentTable = try! device.makeArgumentTable(descriptor: executeSimulationArgumentTableDescriptor)
        executeSimulationArgumentTable.setAddress(interestingTileListBuffer.gpuAddress, index: 0)
        executeSimulationArgumentTable.setAddress(currentGridBuffer.gpuAddress, index: 1)
        executeSimulationArgumentTable.setAddress(nextGridBuffer.gpuAddress, index: 2)
        executeSimulationArgumentTable.setAddress(gridSizePixelsBuffer.gpuAddress, index: 3)
        executeSimulationArgumentTable.setAddress(gridSizeTilesBuffer.gpuAddress, index: 4)

        // (5) create a shared residency set for the buffers used in the classify and simulation kernels
        let residencySet = try! device.makeResidencySet(descriptor: .init())
        residencySet.addAllocation(argumentBuffer)
        residencySet.addAllocation(interestingTileCountBuffer)
        residencySet.addAllocation(interestingTileListBuffer)
        residencySet.addAllocation(currentGridBuffer)
        residencySet.addAllocation(nextGridBuffer)
        residencySet.addAllocation(gridSizePixelsBuffer)
        residencySet.addAllocation(gridSizeTilesBuffer)
        residencySet.addAllocation(completedClassifyThreadgroupCountBuffer)
        residencySet.commit()
        
        // (6) create shared commandQueue, commandBuffer, and commandAllocator that are reused between time steps of the simulation
        commandQueue = device.makeMTL4CommandQueue()!
        commandQueue.addResidencySet(residencySet)
        commandBuffer = device.makeCommandBuffer()!
        commandAllocator = device.makeCommandAllocator()!
    }
    
    private func tileBasedSparse() async {
        // (1) create a compute command encoder for the `classifyTiles` kernel
        commandBuffer.beginCommandBuffer(allocator: commandAllocator, options: .init())
        let classifyTilesComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
        classifyTilesComputeCommandEncoder.setComputePipelineState(classifyTilesComputePipelineState)
        classifyTilesComputeCommandEncoder.setArgumentTable(classifyTilesArgumentTable)
        
        // (2) encode dispatch on CPU, one thread per tile
        let threadsPerThreadgroup = MTLSizeMake(32, 1, 1)
        let threadsPerGrid = MTLSizeMake(tileCountX * tileCountY, 1, 1)
        classifyTilesComputeCommandEncoder.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        classifyTilesComputeCommandEncoder.endEncoding()
        commandBuffer.endCommandBuffer()
        
        // (3) wait for the `classifyTiles` kernel to finish
        try! await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            let commitOptions = MTL4CommitOptions()
            commitOptions.addFeedbackHandler { feedback in
                if let error = feedback.error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume()
                }
            }
            commandQueue.commit([commandBuffer], options: commitOptions)
        }
        
        if interestingTileCount < 1 {
            print("all cells died")
            return
        }
        
        // (4) create a compute command encoder for the `executeSimulation` kernel
        commandBuffer.beginCommandBuffer(allocator: commandAllocator, options: .init())
        let executeSimulationComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
        executeSimulationComputeCommandEncoder.setComputePipelineState(executeSimulationComputePipelineState)
        executeSimulationComputeCommandEncoder.setArgumentTable(executeSimulationArgumentTable)
        
        // (5) encode dispatch on CPU, one threadgroup per interesting tile, tileWidth * tileHeight threads per threadgroup, one thread per cell
        let threadgroupsPerGridX = Int(ceil(sqrt(Double(interestingTileCount))))
        let threadgroupsPerGridY = (Int(interestingTileCount) + threadgroupsPerGridX - 1) / threadgroupsPerGridX
        let threadgroupsPerGrid2 = MTLSizeMake(threadgroupsPerGridX, threadgroupsPerGridY, 1)
        let threadsPerThreadgroup2 = MTLSizeMake(tileWidth, tileHeight, 1)
        executeSimulationComputeCommandEncoder.dispatchThreadgroups(threadgroupsPerGrid: threadgroupsPerGrid2, threadsPerThreadgroup: threadsPerThreadgroup2)
        executeSimulationComputeCommandEncoder.endEncoding()
        commandBuffer.endCommandBuffer()
        
        // (6) wait for the `executeSimulation` kernel to finish
        try! await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            let commitOptions = MTL4CommitOptions()
            commitOptions.addFeedbackHandler { feedback in
                if let error = feedback.error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume()
                }
            }
            
            commandQueue.commit([commandBuffer], options: commitOptions)
        }
        
        // (7) swap the current grid and next grid pointers
        swap(&nextGridBuffer, &currentGridBuffer)

        // (6) rebind the grid buffers
        classifyTilesArgumentTable.setAddress(currentGridBuffer.gpuAddress, index: 2)
        executeSimulationArgumentTable.setAddress(currentGridBuffer.gpuAddress, index: 1)
        executeSimulationArgumentTable.setAddress(nextGridBuffer.gpuAddress, index: 2)

        // (9) reset the interesting tile count to 0
        interestingTileCount = 0
    }

    private func tileBasedSparseICB() async {
        commandBuffer.beginCommandBuffer(allocator: commandAllocator, options: .init())
        
        // (1) create a compute command encoder for the `classifyTilesICB` kernel
        let classifyTilesComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
        classifyTilesComputeCommandEncoder.setComputePipelineState(classifyTilesICBComputePipelineState)
        classifyTilesComputeCommandEncoder.setArgumentTable(classifyTilesICBArgumentTable)

        // (2) block dispatch stage in simulation kernel from running until dispatch stage in tile classification kernel runs
        classifyTilesComputeCommandEncoder.barrier(afterStages: .dispatch, beforeQueueStages: .dispatch)

        // (3) encode dispatch on CPU, one thread per tile
        let threadsPerThreadgroup = MTLSizeMake(32, 1, 1)
        let threadsPerGrid = MTLSizeMake(tileCountX * tileCountY, 1, 1)
        classifyTilesComputeCommandEncoder.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        classifyTilesComputeCommandEncoder.endEncoding()
        
        // (4) create a compute command encoder for the `executeSimulation` kernel
        let executeSimulationComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
        executeSimulationComputeCommandEncoder.setComputePipelineState(executeSimulationICBComputePipelineState)
        let commandRange = 0..<1
        
        // (5) encode the indirect command buffer compute command to `commandBuffer`
        executeSimulationComputeCommandEncoder.executeCommands(buffer: indirectCommandBuffer, range: commandRange)
        executeSimulationComputeCommandEncoder.endEncoding()
        commandBuffer.endCommandBuffer()
        
        // (6) wait for the both kernels to finish
        try! await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            let commitOptions = MTL4CommitOptions()
            commitOptions.addFeedbackHandler { feedback in
                if let error = feedback.error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume()
                }
            }
            
            commandQueue.commit([commandBuffer], options: commitOptions)
        }
        
        // (7) swap the current grid and next grid pointers
        swap(&nextGridBuffer, &currentGridBuffer)

        // (8) rebind the grid buffers for the classify kernel, the rebinding for the simulation kernel is done on the GPU
        classifyTilesICBArgumentTable.setAddress(currentGridBuffer.gpuAddress, index: 3)
        classifyTilesICBArgumentTable.setAddress(nextGridBuffer.gpuAddress, index: 4)

        // (9) reset the interesting tile count to 0
        interestingTileCount = 0

        // (10) reset classify kernel completed threadgroup count to 0
        completedClassifyThreadgroupCount = 0
    }
        
    func start() async {
        running = true
        stepCount = 0
        averageTime = 0.0
        await step()
    }
    
    func stop() {
        running = false
    }
    
    func step() async {
        if !running {
            return
        }
        let startTime = CFAbsoluteTimeGetCurrent()
        await tileBasedSparseICB()
        let time = CFAbsoluteTimeGetCurrent() - startTime
        stepCount += 1
        averageTime = averageTime * (Double(stepCount - 1) / Double(stepCount)) + (time / Double(stepCount))
        if stepCount % 100 == 0 {
            print("Game of Life step \(stepCount) average time to run step: \(averageTime)s")
        }
    }
}

func gameOfLifeLoop() async -> GameOfLife {
    let intervalMS = 2
    let gameOfLife = await GameOfLife()
    
    Task { [weak gameOfLife] in
        let ns = UInt64(intervalMS) * 1_000_000
        
        while !Task.isCancelled {
            try? await Task.sleep(nanoseconds: ns)
            if let gol = gameOfLife {
                if !gol.running {
                    break
                }
                await gol.step()
            } else {
                break
            }
        }
        
        if let gol = gameOfLife {
            gol.stop()
        }
    }
    
    return gameOfLife
}
