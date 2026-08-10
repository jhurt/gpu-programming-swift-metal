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

private func sequential(A: Matrix, B: Matrix) -> Matrix {
    var C = Matrix(rows: A.rows, cols: B.cols, repeating: .zero)
    for row in 0..<A.rows {
        for col in 0..<B.cols {
            var c: Float = .zero
            for k in 0..<A.cols {
                c += A[row, k] * B[k, col]
            }
            C[row, col] = c
        }
    }
    return C
}

func parallel2DGrid(A: inout Matrix, B: inout Matrix) async -> Matrix {
    let processor = ParallelProcessor(log: true)
    
    // (1) define Metal buffers for A, B, and C
    let bufferA = A.toMetalBuffer(device: processor.device)
    let bufferB = B.toMetalBuffer(device: processor.device)
    let cCount = A.rows * B.cols
    let bufferC = processor.device.makeBuffer(length: MemoryLayout<Float>.stride * cCount, options: .storageModeShared)!
    
    // (2) define function constants m, n, and p
    var m = UInt32(A.rows)
    var n = UInt32(A.cols)
    var p = UInt32(B.cols)
    let functionConstantValues = MTLFunctionConstantValues()
    functionConstantValues.setConstantValue(&m, type: .uint, index: 0)
    functionConstantValues.setConstantValue(&n, type: .uint, index: 1)
    functionConstantValues.setConstantValue(&p, type: .uint, index: 2)
    
    // (3) dispatch A.rows * B.cols total threads on a 2D grid
    let threadsPerGrid = MTLSizeMake(A.rows, B.cols, 1)
    let options = ProcessOptions(threadsPerGridMTLSize: threadsPerGrid)
    
    // (4) wait for the result to be written to bufferC
    await processor.process(kernelName: "matrixMultiply2DGrid",
                            buffers: bufferA, bufferB, bufferC,
                            functionConstantValues: functionConstantValues,
                            options: options)
    
    // (5) bind the data in bufferC to a Swift Array and create a C matrix
    let cPtr = bufferC.contents().bindMemory(to: Float.self, capacity: cCount)
    let cData = Array(UnsafeBufferPointer(start: cPtr, count: cCount))
    let C = Matrix(rows: A.rows, cols: B.cols, data: cData)
    return C
}

private func parallel2DGridStride(A: inout Matrix, B: inout Matrix) async -> Matrix {
    let processor = ParallelProcessor(log: true)
    
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
    
    // (1) dispatch a fixed number of threads on a 2D grid
    let threadsPerGrid = MTLSizeMake(min(1024, A.rows), min(1024, B.cols), 1)
    let options = ProcessOptions(threadsPerGridMTLSize: threadsPerGrid)
    
    await processor.process(kernelName: "matrixMultiply2DGridStride",
                            buffers: bufferA, bufferB, bufferC,
                            functionConstantValues: functionConstantValues,
                            options: options)
    
    let cPtr = bufferC.contents().bindMemory(to: Float.self, capacity: cCount)
    let cData = Array(UnsafeBufferPointer(start: cPtr, count: cCount))
    let C = Matrix(rows: A.rows, cols: B.cols, data: cData)
    return C
}

private func parallel3DGrid(A: inout Matrix, B: inout Matrix) async -> Matrix {
    let processor = ParallelProcessor(log: true)
    
    // (1) define Metal buffers for A, B, and C
    let bufferA = A.toMetalBuffer(device: processor.device)
    let bufferB = B.toMetalBuffer(device: processor.device)
    let cCount = A.rows * B.cols
    let bufferC = processor.device.makeBuffer(length: MemoryLayout<Float>.stride * cCount, options: .storageModeShared)!
    
    // (2) define function constants m, n, and p
    var m = UInt32(A.rows)
    var n = UInt32(A.cols)
    var p = UInt32(B.cols)
    let functionConstantValues = MTLFunctionConstantValues()
    functionConstantValues.setConstantValue(&m, type: .uint, index: 0)
    functionConstantValues.setConstantValue(&n, type: .uint, index: 1)
    functionConstantValues.setConstantValue(&p, type: .uint, index: 2)
    
    // (3) dispatch A.rows * B.cols * A.cols total threads on a 3D grid
    let threadsPerGrid = MTLSizeMake(A.rows, B.cols, A.cols)
    let options = ProcessOptions(threadsPerGridMTLSize: threadsPerGrid)
    
    // (4) wait for the result to be written to bufferC
    await processor.process(kernelName: "matrixMultiply3DGrid",
                            buffers: bufferA, bufferB, bufferC,
                            functionConstantValues: functionConstantValues,
                            options: options)
    
    // (5) bind the data in bufferC to a Swift Array and create a C matrix
    let cPtr = bufferC.contents().bindMemory(to: Float.self, capacity: cCount)
    let cData = Array(UnsafeBufferPointer(start: cPtr, count: cCount))
    let C = Matrix(rows: A.rows, cols: B.cols, data: cData)
    return C
}

private func parallelTiledThreadgroupMemory(A: inout Matrix, B: inout Matrix) async -> Matrix {
    let processor = ParallelProcessor(log: true)
    
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
    
    let tileSize = 16
    let threadsPerGrid = MTLSizeMake(B.cols, A.rows, 1)
    let threadsPerThreadgroup = MTLSizeMake(tileSize, tileSize, 1)
    let options = ProcessOptions(threadsPerGridMTLSize: threadsPerGrid, threadsPerThreadgroupMTLSize: threadsPerThreadgroup)
    await processor.process(kernelName: "matrixMultiplyTiledThreadgroupMemory",
                            buffers: bufferA, bufferB, bufferC,
                            functionConstantValues: functionConstantValues,
                            options: options)
    
    let cPtr = bufferC.contents().bindMemory(to: Float.self, capacity: cCount)
    let cData = Array(UnsafeBufferPointer(start: cPtr, count: cCount))
    let C = Matrix(rows: A.rows, cols: B.cols, data: cData)
    return C
}

private func parallelSimdSingleThreadgroup(A: inout Matrix, B: inout Matrix) async -> Matrix {
    let processor = ParallelProcessor(log: true)
    
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
    
    let threadgroupsPerGrid = MTLSizeMake(1, 1, 1)
    let threadsPerThreadgroup = MTLSizeMake(32, 1, 1)
    let options = ProcessOptions(threadgroupsPerGridMTLSize: threadgroupsPerGrid, threadsPerThreadgroupMTLSize: threadsPerThreadgroup)
    await processor.process(kernelName: "matrixMultiplySimdSingleThreadgroup",
                            buffers: bufferA, bufferB, bufferC,
                            functionConstantValues: functionConstantValues,
                            options: options)
    
    let cPtr = bufferC.contents().bindMemory(to: Float.self, capacity: cCount)
    let cData = Array(UnsafeBufferPointer(start: cPtr, count: cCount))
    let C = Matrix(rows: A.rows, cols: B.cols, data: cData)
    return C
}

private func parallelSimdMultiThreadgroup1(A: inout Matrix, B: inout Matrix) async -> Matrix {
    let processor = ParallelProcessor(log: true)
    
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
    
    let threadgroupsPerGrid = MTLSizeMake(B.cols / 8, A.rows / 8, 1)
    let threadsPerThreadgroup = MTLSizeMake(32, 1, 1)
    let options = ProcessOptions(threadgroupsPerGridMTLSize: threadgroupsPerGrid, threadsPerThreadgroupMTLSize: threadsPerThreadgroup)
    await processor.process(kernelName: "matrixMultiplySimdMultiThreadgroup1",
                            buffers: bufferA, bufferB, bufferC,
                            functionConstantValues: functionConstantValues,
                            options: options)
    
    let cPtr = bufferC.contents().bindMemory(to: Float.self, capacity: cCount)
    let cData = Array(UnsafeBufferPointer(start: cPtr, count: cCount))
    let C = Matrix(rows: A.rows, cols: B.cols, data: cData)
    return C
}

private func parallelSimdMultiThreadgroup2(A: inout Matrix, B: inout Matrix) async -> Matrix {
    let kernelName = "matrixMultiplySimdMultiThreadgroup2"
    let processor = ParallelProcessor(log: true)
    
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
    
    let simdGroupsPerThreadgroup = 2;
    let threadgroupsPerGrid = MTLSizeMake(B.cols / 8, A.rows / (simdGroupsPerThreadgroup * 8), 1)
    let threadsPerThreadgroup = MTLSizeMake(32, simdGroupsPerThreadgroup, 1)
    let options = ProcessOptions(threadgroupsPerGridMTLSize: threadgroupsPerGrid, threadsPerThreadgroupMTLSize: threadsPerThreadgroup)
    await processor.process(kernelName: kernelName,
                            buffers: bufferA, bufferB, bufferC,
                            functionConstantValues: functionConstantValues,
                            options: options)
    
    let cPtr = bufferC.contents().bindMemory(to: Float.self, capacity: cCount)
    let cData = Array(UnsafeBufferPointer(start: cPtr, count: cCount))
    let C = Matrix(rows: A.rows, cols: B.cols, data: cData)
    return C
}

private func parallelSimd4x4(A: inout Matrix, B: inout Matrix) async -> Matrix {
    let processor = ParallelProcessor(log: true)
    
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
    
    let simdGroupsPerThreadgroup = 8;
    let threadgroupsPerGrid = MTLSizeMake(B.cols / 32, A.rows / (simdGroupsPerThreadgroup * 32), 1)
    let threadsPerThreadgroup = MTLSizeMake(32, simdGroupsPerThreadgroup, 1)
    let options = ProcessOptions(threadgroupsPerGridMTLSize: threadgroupsPerGrid, threadsPerThreadgroupMTLSize: threadsPerThreadgroup)
    await processor.process(kernelName: "matrixMultiplySimd4x4",
                            buffers: bufferA, bufferB, bufferC,
                            functionConstantValues: functionConstantValues,
                            options: options)
    
    let cPtr = bufferC.contents().bindMemory(to: Float.self, capacity: cCount)
    let cData = Array(UnsafeBufferPointer(start: cPtr, count: cCount))
    let C = Matrix(rows: A.rows, cols: B.cols, data: cData)
    return C
}

private func parallelSimd4x4ThreadgroupMemoryB(A: inout Matrix, B: inout Matrix) async -> Matrix {
    let processor = ParallelProcessor(log: true)
    
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
    
    let simdGroupsPerThreadgroup = 4;
    let threadgroupsPerGrid = MTLSizeMake(B.cols / 32, A.rows / (simdGroupsPerThreadgroup * 32), 1)
    let threadsPerThreadgroup = MTLSizeMake(32, simdGroupsPerThreadgroup, 1)
    let options = ProcessOptions(threadgroupsPerGridMTLSize: threadgroupsPerGrid, threadsPerThreadgroupMTLSize: threadsPerThreadgroup)
    await processor.process(kernelName: "matrixMultiplySimd4x4ThreadgroupMemoryB",
                            buffers: bufferA, bufferB, bufferC,
                            functionConstantValues: functionConstantValues,
                            options: options)
    
    let cPtr = bufferC.contents().bindMemory(to: Float.self, capacity: cCount)
    let cData = Array(UnsafeBufferPointer(start: cPtr, count: cCount))
    let C = Matrix(rows: A.rows, cols: B.cols, data: cData)
    return C
}

private func parallelSimd4x4ThreadgroupMemoryA(A: inout Matrix, B: inout Matrix) async -> Matrix {
    let processor = ParallelProcessor(log: true)
    
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
    
    let simdGroupsPerThreadgroup = 4;
    let threadgroupsPerGrid = MTLSizeMake(B.cols / (simdGroupsPerThreadgroup * 32), A.rows / 32, 1)
    let threadsPerThreadgroup = MTLSizeMake(32, simdGroupsPerThreadgroup, 1)
    let options = ProcessOptions(threadgroupsPerGridMTLSize: threadgroupsPerGrid, threadsPerThreadgroupMTLSize: threadsPerThreadgroup)
    await processor.process(kernelName: "matrixMultiplySimd4x4ThreadgroupMemoryA",
                            buffers: bufferA, bufferB, bufferC,
                            functionConstantValues: functionConstantValues,
                            options: options)
    
    let cPtr = bufferC.contents().bindMemory(to: Float.self, capacity: cCount)
    let cData = Array(UnsafeBufferPointer(start: cPtr, count: cCount))
    let C = Matrix(rows: A.rows, cols: B.cols, data: cData)
    return C
}

func matrixMultiply(A: inout Matrix, B: inout Matrix) async -> Matrix {
    return await parallelSimd4x4ThreadgroupMemoryA(A: &A, B: &B)
}

private func parallelMPS(A: inout Matrix, B: inout Matrix) -> Matrix {
    let device = MTLCreateSystemDefaultDevice()!
    let matrixA = A.toMPSMatrix(device: device)
    let matrixB = B.toMPSMatrix(device: device)
    let commandQueue = device.makeCommandQueue()!
    let commandBuffer = commandQueue.makeCommandBuffer()!
    
    let rowBytesC = MPSMatrixDescriptor.rowBytes(fromColumns: B.cols, dataType: .float32)
    let matrixC = MPSMatrix(device: device, descriptor: MPSMatrixDescriptor(rows: A.rows, columns: B.cols, rowBytes: rowBytesC, dataType: .float32))
    let matrixMultiplication = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false, resultRows: A.rows, resultColumns: B.cols, interiorColumns: A.cols, alpha: 1.0, beta: 0.0)
    matrixMultiplication.encode(commandBuffer: commandBuffer, leftMatrix: matrixA, rightMatrix: matrixB, resultMatrix: matrixC)
    let startTime = CFAbsoluteTimeGetCurrent()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    print(String(format: "parallelMPS total time: %.4f seconds", CFAbsoluteTimeGetCurrent() - startTime))
    print(String(format: "parallelMPS GPU time: %.4f seconds", commandBuffer.gpuEndTime - commandBuffer.gpuStartTime))
    return Matrix.fromMPSMatrix(matrixC, type: Float.self)
}

func matrixMultiplication() async {
    let shapes: [(Int, Int, Int)] = [
        (1024, 768, 1024),
        (768, 1024, 768),
        (16384, 256, 1024),
        (7680, 10240, 7680),
        (10240, 7680, 10240),
        (20480, 20480, 20480),
        (1024, 65536, 32768),
        (32768, 65536, 1024),
    ]
    
    for shape in shapes {
        let countA = shape.0 * shape.1
        let countB = shape.1 * shape.2
        
        print("===========================")
        print("generating matrices for shape: \(shape.0) x \(shape.1) x \(shape.2)")
        let dataA = await ParallelData.randomFloatArray(length: countA)
        var A = Matrix(rows: shape.0, cols: shape.1, data: dataA)
        let dataB = await ParallelData.randomFloatArray(length: countB)
        var B = Matrix(rows: shape.1, cols: shape.2, data: dataB)
        print("A \(A)")
        print("B \(B)\n")
        
        var parallelMatrixTiledThreadgroupMemory = await parallelTiledThreadgroupMemory(A: &A, B: &B)
        print("parallelTiledThreadgroupMemory result \(parallelMatrixTiledThreadgroupMemory)\n")
        
//        var result = await parallel2DGridStride(A: &A, B: &B)
//        print("parallel2DGridStride result \(result)")
//        await compareMatrices(A: &parallelMatrixTiledThreadgroupMemory, B: &result, labelA: "parallelMatrixTiledThreadgroupMemory", labelB: "parallel2DGridStride")
        
        var result = await parallelSimdMultiThreadgroup1(A: &A, B: &B)
        print("parallelSimdMultiThreadgroup1 result \(result)")
        await compareMatrices(A: &parallelMatrixTiledThreadgroupMemory, B: &result, labelA: "parallelMatrixTiledThreadgroupMemory", labelB: "parallelSimdMultiThreadgroup1")
        
        result = await parallelSimdMultiThreadgroup2(A: &A, B: &B)
        print("parallelSimdMultiThreadgroup2 result \(result)")
        await compareMatrices(A: &parallelMatrixTiledThreadgroupMemory, B: &result, labelA: "parallelMatrixTiledThreadgroupMemory", labelB: "parallelSimdMultiThreadgroup2")
        
        result = await parallelSimd4x4(A: &A, B: &B)
        print("parallelSimd4x4 result \(result)")
        await compareMatrices(A: &parallelMatrixTiledThreadgroupMemory, B: &result, labelA: "parallelMatrixTiledThreadgroupMemory", labelB: "parallelSimd4x4")
        
        result = await parallelSimd4x4ThreadgroupMemoryB(A: &A, B: &B)
        print("parallelSimd4x4ThreadgroupMemoryB result \(result)")
        await compareMatrices(A: &parallelMatrixTiledThreadgroupMemory, B: &result, labelA: "parallelMatrixTiledThreadgroupMemory", labelB: "parallelSimd4x4ThreadgroupMemoryB")
        
        result = await parallelSimd4x4ThreadgroupMemoryA(A: &A, B: &B)
        print("parallelSimd4x4ThreadgroupMemoryA result \(result)")
        await compareMatrices(A: &parallelMatrixTiledThreadgroupMemory, B: &result, labelA: "parallelMatrixTiledThreadgroupMemory", labelB: "parallelSimd4x4ThreadgroupMemoryA")
        
        result = parallelMPS(A: &A, B: &B)
        print("parallelMPS result \(result)")
        await compareMatrices(A: &parallelMatrixTiledThreadgroupMemory, B: &result, labelA: "parallelMatrixTiledThreadgroupMemory", labelB: "parallelMPS")
    }
}
