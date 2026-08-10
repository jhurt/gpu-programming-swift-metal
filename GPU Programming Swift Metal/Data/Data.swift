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

class ParallelData {
    static func randomFloatArray(length: Int) async -> [Float] {
        let processor = ParallelProcessor()
        
        let outputBuffer = processor.device.makeBuffer(length: MemoryLayout<Float>.stride * length, options: .storageModeShared)!
        var countULong = UInt64(length)
        let countBuffer = processor.device.makeBuffer(bytes: &countULong, length: MemoryLayout<UInt64>.stride, options: .storageModeShared)!
        
        let kernelName = "randomFloat"
        let computePipelineState = await processor.registerKernel(kernelName)
        
        // (1) use max total threads per threadgroup as the threads per threadgroup dispatch value
        let threadgroups = 1024
        let threadsPerThreadgroup = computePipelineState.maxTotalThreadsPerThreadgroup
        let threadsPerGrid = threadgroups * threadsPerThreadgroup
        let options = ProcessOptions(threadsPerGrid: threadsPerGrid)
        await processor.process(kernelName: kernelName, buffers: outputBuffer, countBuffer, options: options)
        
        let resultPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: length)
        return Array(UnsafeBufferPointer(start: resultPointer, count: length))
    }

    static func randomBoundedFloatArray(length: Int, lowerBound: Float, upperBound: Float) async -> [Float] {
        let processor = ParallelProcessor()
        
        let outputBuffer = processor.device.makeBuffer(length: MemoryLayout<Float>.stride * length, options: .storageModeShared)!
        var countULong = UInt64(length)
        let countBuffer = processor.device.makeBuffer(bytes: &countULong, length: MemoryLayout<UInt64>.stride, options: .storageModeShared)!
        
        var lb = lowerBound, ub = upperBound
        let lowerBoundBuffer = processor.device.makeBuffer(bytes: &lb, length: MemoryLayout<Float>.stride, options: .storageModeShared)!
        let upperBoundBuffer = processor.device.makeBuffer(bytes: &ub, length: MemoryLayout<Float>.stride, options: .storageModeShared)!
        
        let kernelName = "randomBoundedFloat"
        let computePipelineState = await processor.registerKernel(kernelName)
        
        let threadgroups = 1024
        let threadsPerThreadgroup = computePipelineState.maxTotalThreadsPerThreadgroup
        let threadsPerGrid = threadgroups * threadsPerThreadgroup
        let options = ProcessOptions(threadsPerGrid: threadsPerGrid)
        await processor.process(kernelName: kernelName, buffers: outputBuffer, countBuffer, lowerBoundBuffer, upperBoundBuffer, options: options)
        
        let resultPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: length)
        return Array(UnsafeBufferPointer(start: resultPointer, count: length))
    }

    static func randomInt32Array(length: Int) async -> [Int32] {
        let processor = ParallelProcessor()
        
        let outputBuffer = processor.device.makeBuffer(length: MemoryLayout<Int32>.stride * length, options: .storageModeShared)!
        var countULong = UInt64(length)
        let countBuffer = processor.device.makeBuffer(bytes: &countULong, length: MemoryLayout<UInt64>.stride, options: .storageModeShared)!
        
        let kernelName = "randomInt"
        let computePipelineState = await processor.registerKernel(kernelName)
        
        let threadgroups = 1024
        let threadsPerThreadgroup = computePipelineState.maxTotalThreadsPerThreadgroup
        let threadsPerGrid = threadgroups * threadsPerThreadgroup
        let options = ProcessOptions(threadsPerGrid: threadsPerGrid)
        await processor.process(kernelName: kernelName, buffers: outputBuffer, countBuffer, options: options)
        
        let resultPointer = outputBuffer.contents().bindMemory(to: Int32.self, capacity: length)
        return Array(UnsafeBufferPointer(start: resultPointer, count: length))
    }

    static func floatArrayWithValues(value: Float, length: Int) async -> [Float] {
        let processor = ParallelProcessor()
        
        let outputBuffer = processor.device.makeBuffer(length: MemoryLayout<Float>.stride * length, options: .storageModeShared)!
        var countULong = UInt64(length)
        let countBuffer = processor.device.makeBuffer(bytes: &countULong, length: MemoryLayout<UInt64>.stride, options: .storageModeShared)!
        var initValue = value
        let valueBuffer = processor.device.makeBuffer(bytes: &initValue, length: MemoryLayout<UInt64>.stride, options: .storageModeShared)!

        let kernelName = "floatArrayWithValues"
        let computePipelineState = await processor.registerKernel(kernelName)
        
        // (1) use max total threads per threadgroup as the threads per threadgroup dispatch value
        let threadgroups = 1024
        let threadsPerThreadgroup = computePipelineState.maxTotalThreadsPerThreadgroup
        let threadsPerGrid = threadgroups * threadsPerThreadgroup
        let options = ProcessOptions(threadsPerGrid: threadsPerGrid)
        await processor.process(kernelName: kernelName, buffers: outputBuffer, countBuffer, valueBuffer, options: options)
        
        let resultPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: length)
        return Array(UnsafeBufferPointer(start: resultPointer, count: length))
    }

    static func randomPoint2DArray(length: Int, minX: inout Int32, maxX: inout Int32, minY: inout Int32, maxY: inout Int32) async -> [Point2D] {
        let processor = ParallelProcessor()
        let kernelName = "randomPoint2D"
        
        let outputBuffer = processor.device.makeBuffer(length: MemoryLayout<Point2D>.stride * length, options: .storageModeShared)!
        var countU32 = UInt32(length)
        let countBuffer = processor.device.makeBuffer(bytes: &countU32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        
        let minXBuffer = processor.device.makeBuffer(bytes: &minX, length: MemoryLayout<Int32>.stride, options: .storageModeShared)!
        let maxXBuffer = processor.device.makeBuffer(bytes: &maxX, length: MemoryLayout<Int32>.stride, options: .storageModeShared)!
        let minYBuffer = processor.device.makeBuffer(bytes: &minY, length: MemoryLayout<Int32>.stride, options: .storageModeShared)!
        let maxYBuffer = processor.device.makeBuffer(bytes: &maxY, length: MemoryLayout<Int32>.stride, options: .storageModeShared)!
        
        let options = ProcessOptions(threadsPerGrid: length)
        await processor.process(kernelName: kernelName, buffers: outputBuffer, countBuffer, minXBuffer, maxXBuffer, minYBuffer, maxYBuffer, options: options)
        
        let resultPointer = outputBuffer.contents().bindMemory(to: Point2D.self, capacity: length)
        return Array(UnsafeBufferPointer(start: resultPointer, count: length))
    }
    
    static func newULongArray(length: Int) async -> [UInt64] {
        let processor = ParallelProcessor()
        
        let outputBuffer = processor.device.makeBuffer(length: MemoryLayout<UInt64>.stride * length, options: .storageModeShared)!
        var countULong = UInt64(length)
        let countBuffer = processor.device.makeBuffer(bytes: &countULong, length: MemoryLayout<UInt64>.stride, options: .storageModeShared)!
        
        let kernelName = "newULong"
        let computePipelineState = await processor.registerKernel(kernelName)
        
        let threadgroups = 1024
        let threadsPerThreadgroup = computePipelineState.maxTotalThreadsPerThreadgroup
        let threadsPerGrid = threadgroups * threadsPerThreadgroup
        let options = ProcessOptions(threadsPerGrid: threadsPerGrid)
        await processor.process(kernelName: kernelName, buffers: outputBuffer, countBuffer, options: options)
        
        let resultPointer = outputBuffer.contents().bindMemory(to: UInt64.self, capacity: length)
        return Array(UnsafeBufferPointer(start: resultPointer, count: length))
    }
    
    static func compareFloatArrays(A: inout [Float], B: inout [Float]) async -> (UInt32, Float) {
        if A.count != B.count {
            fatalError("Arrays must have same length \(A.count) != \(B.count)")
        }
        
        let processor = ParallelProcessor()
        
        let bufferA = processor.device.makeBuffer(bytesNoCopy: &A, length: MemoryLayout<Float>.stride * A.count, options: .storageModeShared)!
        let bufferB = processor.device.makeBuffer(bytesNoCopy: &B, length: MemoryLayout<Float>.stride * B.count, options: .storageModeShared)!

        var countULong = UInt64(A.count)
        let countBuffer = processor.device.makeBuffer(bytes: &countULong, length: MemoryLayout<UInt64>.stride, options: .storageModeShared)!

        var numberOfDiffs = UInt32(0)
        let numberOfDiffsBuffer = processor.device.makeBuffer(bytesNoCopy: &numberOfDiffs, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!

        var largestDiff = Float(0.0)
        let largestDiffBuffer = processor.device.makeBuffer(bytesNoCopy: &largestDiff, length: MemoryLayout<Float>.stride, options: .storageModeShared)!
        
        let kernelName = "compareFloatArrays"
        let computePipelineState = await processor.registerKernel(kernelName)
        
        let threadgroups = 1024
        let threadsPerThreadgroup = computePipelineState.maxTotalThreadsPerThreadgroup
        let threadsPerGrid = threadgroups * threadsPerThreadgroup
        let options = ProcessOptions(threadsPerGrid: threadsPerGrid)
        await processor.process(kernelName: kernelName, buffers: bufferA, bufferB, countBuffer, numberOfDiffsBuffer, largestDiffBuffer, options: options)
        
        return (numberOfDiffs, largestDiff)
    }
}
