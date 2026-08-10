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
import MetalPerformanceShaders

private func parallelTranspose(_ input: inout Matrix) async -> Matrix {
    let processor = ParallelProcessor()

    // (1) register the kernel and read the max threads per threadgroup
    let kernelName = "matrixTranspose"
    let computePipelineState = await processor.registerKernel(kernelName)
    let maxThreadsPerThreadgroup = computePipelineState.maxTotalThreadsPerThreadgroup

    // (2) base the tile size on the max threads per threadgroup
    var tileSize: UInt32;
    if maxThreadsPerThreadgroup == 1024 {
        tileSize = 32;
    } else {
        tileSize = 16;
    }
    
    // (3) the threadgroup memory will hold a tileSize x tileSize float32 tile
    let threadgroupMemory = Int(tileSize) * Int(tileSize) * MemoryLayout<Float>.stride
    
    let inputBuffer = input.toMetalBuffer(device: processor.device)
    var cols = UInt32(input.cols)
    let colsBuffer = processor.device.makeBuffer(bytes: &cols, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    var rows = UInt32(input.rows)
    let rowsBuffer = processor.device.makeBuffer(bytes: &rows, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    let tileSizeBuffer = processor.device.makeBuffer(bytes: &tileSize, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    let outputBuffer = processor.device.makeBuffer(length: MemoryLayout<Float>.stride * input.rows * input.cols, options: .storageModeShared)!
    
    // (4) dispatch input.cols x input.rows threads, each with a threadgroup size of tileSize x tileSize
    await processor.process(kernelName: kernelName,
                            buffers: inputBuffer, colsBuffer, rowsBuffer, tileSizeBuffer, outputBuffer,
                            threadgroupMemory: threadgroupMemory,
                            options: ProcessOptions(
                                threadsPerGridMTLSize: MTLSizeMake(input.cols, input.rows, 1),
                                threadsPerThreadgroupMTLSize: MTLSizeMake(Int(tileSize), Int(tileSize), 1)))
    
    let ptr = outputBuffer.contents().bindMemory(to: Float.self, capacity: input.rows * input.cols)
    let data = Array(UnsafeBufferPointer(start: ptr, count: input.rows * input.cols))
    return Matrix(rows: input.cols, cols: input.rows, data: data)
}

func matrixTranspose(_ input: inout Matrix) async -> Matrix {
    return await parallelTranspose(&input)
}

private func parallelMPS(_ input: inout Matrix) -> Matrix {
    let device = MTLCreateSystemDefaultDevice()!
    let matrix = input.toMPSMatrix(device: device)
    
    let transposedDesctiptor = MPSMatrixDescriptor(
        rows: matrix.columns,
        columns: matrix.rows,
        rowBytes: matrix.rows * MemoryLayout<Float>.size,
        dataType: .float32
    )
    let transposedMatrix = MPSMatrix(device: device, descriptor: transposedDesctiptor)
    
    let copyDescriptor = MPSMatrixCopyDescriptor(
        sourceMatrix: matrix,
        destinationMatrix: transposedMatrix,
        offsets: .init()
    )
    
    let copy = MPSMatrixCopy(
        device: device,
        copyRows: matrix.rows,
        copyColumns: matrix.columns,
        sourcesAreTransposed: false,
        destinationsAreTransposed: true
    )
    
    let commandQueue = device.makeCommandQueue()!
    let commandBuffer = commandQueue.makeCommandBuffer()!
    copy.encode(commandBuffer: commandBuffer, copyDescriptor: copyDescriptor)
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    print(String(format: "parallelMPS GPU time: %.4f seconds", commandBuffer.gpuEndTime - commandBuffer.gpuStartTime))
    
    return Matrix.fromMPSMatrix(transposedMatrix, type: Float.self)
}

func compareMatrices(A: inout Matrix, B: inout Matrix, labelA: String, labelB: String) async {
    let (numberOfDiffs, largestDiff) = await ParallelData.compareFloatArrays(A: &A.data, B: &B.data)
    if numberOfDiffs > 0 {
        fatalError("found \(numberOfDiffs) diffs between \(labelA) and \(labelB), largest: \(largestDiff)")
    } else {
        print("found no diffs between \(labelA) and \(labelB)\n")
    }
}

func compareMatricesRanges(A: Matrix, B: Matrix) {
    var start = -1
    var differences = 0
    for i in 0..<A.data.count {
        let x = A.data[i]
        let y = B.data[i]
        if abs(x - y) > 1e-4 {
            print("x != y at index \(i) (x: \(x), y: \(y)")
            
            differences += 1
            if differences > 100 {
                return
            }
            
            if start == -1 {
                start = i
            }
        } else {
            if start != -1 {
                print("matrices differ in range \(start)..<\(i)")
            }
            start = -1
        }
    }
    if start != -1 {
        print("matrices differ in range \(start)..<\(A.data.count)")
    }
}

func matrixTranspose() async {
    let shapes: [(Int, Int)] = [
        (1024, 768),
        (768, 1024),
        (16384, 256),
        (7680, 10240),
        (10240, 7680),
        (20480, 20480),
        (1024, 65536),
        (32768, 65536),
    ]
    
    for shape in shapes {
        let count = shape.0 * shape.1
        
        print("===========================")
        print("generating matrix for shape: \(shape.0) x \(shape.1)")
        let data = await ParallelData.randomFloatArray(length: count)
        var A = Matrix(rows: shape.0, cols: shape.1, data: data)
        var startTime = CFAbsoluteTimeGetCurrent()
        var transposedA = await parallelTranspose(&A)
        print(String(format: "parallelTranspose %.4f seconds", CFAbsoluteTimeGetCurrent() - startTime))
        
        startTime = CFAbsoluteTimeGetCurrent()
        var transposedB = parallelMPS(&A)
        print(String(format: "parallelMPS %.4f seconds", CFAbsoluteTimeGetCurrent() - startTime))
        
        await compareMatrices(A: &transposedA, B: &transposedB, labelA: "parallelTranspose", labelB: "parallelMPS")
    }
}
