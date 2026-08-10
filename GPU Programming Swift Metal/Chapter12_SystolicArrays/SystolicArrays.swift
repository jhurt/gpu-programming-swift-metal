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

private func gemm2x2Threadgroup(A: inout Matrix, B: inout Matrix) async -> Matrix {
    let processor = ParallelProcessor()
    
    let bufferA = A.toMetalBuffer(device: processor.device)
    let bufferB = B.toMetalBuffer(device: processor.device)
    let cCount = A.rows * B.cols
    let bufferC = processor.device.makeBuffer(length: MemoryLayout<Float>.stride * cCount, options: .storageModeShared)!
    
    let options = ProcessOptions(threadsPerGrid: 4)
    await processor.process(kernelName: "systolicArrayGemm2x2Threadgroup", buffers: bufferA, bufferB, bufferC, options: options)
    
    let cPtr = bufferC.contents().bindMemory(to: Float.self, capacity: cCount)
    let cData = Array(UnsafeBufferPointer(start: cPtr, count: cCount))
    let C = Matrix(rows: A.rows, cols: B.cols, data: cData)
    return C
}

private func gemmThreadgroup(A: inout Matrix, B: inout Matrix) async -> Matrix {
    let processor = ParallelProcessor()
    
    let bufferA = A.toMetalBuffer(device: processor.device)
    let bufferB = B.toMetalBuffer(device: processor.device)
    let cCount = A.rows * B.cols
    let bufferC = processor.device.makeBuffer(length: MemoryLayout<Float>.stride * cCount, options: .storageModeShared)!
    
    var m = UInt32(A.rows)
    var n = UInt32(A.cols)
    var p = UInt32(B.cols)
    let functionConstantValues = MTLFunctionConstantValues()
    functionConstantValues.setConstantValue(&m, type: .uint, index: 0)
    functionConstantValues.setConstantValue(&n, type: .uint, index: 1)
    functionConstantValues.setConstantValue(&p, type: .uint, index: 2)
    
    let kernelName = "systolicArrayGemmThreadgroup"
    let computePipelineState = await processor.registerKernel(kernelName, functionConstantValues: functionConstantValues)
    
    // (1) determine how many PE's are needed
    let totalPEs = A.rows * B.cols
    // (2) fail if the number of PE's exceeds the pipeline's max threads per threadgroup
    if totalPEs > computePipelineState.maxTotalThreadsPerThreadgroup {
        fatalError("input matrices exceed maximum supported size")
    }
    let options = ProcessOptions(threadsPerGrid: totalPEs, threadsPerThreadgroup: totalPEs)
    // (3) determine threadgroup memory for each PE's A and B
    let threadgroupMemoryA = totalPEs * MemoryLayout<Float>.stride
    let threadgroupMemoryB = totalPEs * MemoryLayout<Float>.stride
    await processor.process(kernelName: kernelName, buffers: bufferA, bufferB, bufferC, threadgroupMemory: threadgroupMemoryA, threadgroupMemoryB, functionConstantValues: functionConstantValues, options: options)
    
    let cPtr = bufferC.contents().bindMemory(to: Float.self, capacity: cCount)
    let cData = Array(UnsafeBufferPointer(start: cPtr, count: cCount))
    let C = Matrix(rows: A.rows, cols: B.cols, data: cData)
    return C
}

private func gemmSIMD(A: inout Matrix, B: inout Matrix) async -> Matrix {
    let processor = ParallelProcessor()
    
    let bufferA = A.toMetalBuffer(device: processor.device)
    let bufferB = B.toMetalBuffer(device: processor.device)
    let cCount = A.rows * B.cols
    let bufferC = processor.device.makeBuffer(length: MemoryLayout<Float>.stride * cCount, options: .storageModeShared)!
    
    var m = UInt32(A.rows)
    var n = UInt32(A.cols)
    var p = UInt32(B.cols)
    let functionConstantValues = MTLFunctionConstantValues()
    functionConstantValues.setConstantValue(&m, type: .uint, index: 0)
    functionConstantValues.setConstantValue(&n, type: .uint, index: 1)
    functionConstantValues.setConstantValue(&p, type: .uint, index: 2)
    
    let kernelName = "systolicArrayGemmSIMD"
    let computePipelineState = await processor.registerKernel(kernelName, functionConstantValues: functionConstantValues)

    let totalPEs = A.rows * B.cols
    // (1) fail if the number of PE's exceeds the pipeline's thread execution width
    if totalPEs > computePipelineState.threadExecutionWidth {
        fatalError("input matrices exceed maximum supported size")
    }
    let options = ProcessOptions(threadsPerGrid: totalPEs, threadsPerThreadgroup: totalPEs)
    await processor.process(kernelName: kernelName, buffers: bufferA, bufferB, bufferC, functionConstantValues: functionConstantValues, options: options)
    
    let cPtr = bufferC.contents().bindMemory(to: Float.self, capacity: cCount)
    let cData = Array(UnsafeBufferPointer(start: cPtr, count: cCount))
    let C = Matrix(rows: A.rows, cols: B.cols, data: cData)
    return C
}

private func generateData(start: Float, count: Int) -> [Float] {
    return (0..<count).map { i in
        start + Float(i)
    }
}

private func threadgroupSystolicArrayGemm() async {
    let shapes: [(Int, Int, Int)] = [
        (2, 2, 2),
        (4, 4, 4),
        (4, 8, 4),
        (4, 8, 2),
        (8, 4, 2),
        (8, 4, 8),
    ]
    for shape in shapes {
        var A = Matrix(rows: shape.0, cols: shape.1, data: generateData(start: 1.0, count: shape.0 * shape.1))
        var B = Matrix(rows: shape.1, cols: shape.2, data: generateData(start: 1.0 + Float(shape.0 * shape.1), count: shape.1 * shape.2))
        print("A \(A)")
        print("B \(B)\n")
        
        var C1 = await parallel2DGrid(A: &A, B: &B)
        print("parallel2DGrid result \(C1)\n")
        
        var C2 = await gemmThreadgroup(A: &A, B: &B)
        print("gemmThreadgroup result \(C2)\n")
        await compareMatrices(A: &C1, B: &C2, labelA: "parallel2DGrid", labelB: "gemmThreadgroup")
    }
}

private func simdSystolicArrayGemm() async {
    let shapes: [(Int, Int, Int)] = [
        (2, 2, 2),
        (4, 4, 4),
        (4, 8, 4),
        (4, 8, 2),
        (8, 4, 2),
    ]
    for shape in shapes {
        var A = Matrix(rows: shape.0, cols: shape.1, data: generateData(start: 1.0, count: shape.0 * shape.1))
        var B = Matrix(rows: shape.1, cols: shape.2, data: generateData(start: 1.0 + Float(shape.0 * shape.1), count: shape.1 * shape.2))
        print("A \(A)")
        print("B \(B)\n")
        
        var C1 = await parallel2DGrid(A: &A, B: &B)
        print("parallel2DGrid result \(C1)\n")
        
        var C2 = await gemmSIMD(A: &A, B: &B)
        print("gemmSIMD result \(C2)\n")
        await compareMatrices(A: &C1, B: &C2, labelA: "parallel2DGrid", labelB: "gemmSIMD")
    }
}

func systolicArrays() async {
    await threadgroupSystolicArrayGemm()
    await simdSystolicArrayGemm()
}
