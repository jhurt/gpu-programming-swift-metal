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

private func multiHeadAttentionTensor(sequenceLength:Int,
                                      embeddingDimension: Int,
                                      headCount: Int,
                                      embeddingsData: inout [Float],
                                      queryWeightData: inout [Float],
                                      keyWeightData: inout [Float],
                                      valueWeightData: inout [Float]) async -> Matrix {
    let startTime = CFAbsoluteTimeGetCurrent()
    
    let device = MTLCreateSystemDefaultDevice()!
    
    let headDimension = embeddingDimension / headCount
    
    // embeddings tensor
    let tensorDescriptorEmbeddings = MTLTensorDescriptor()
    tensorDescriptorEmbeddings.usage = .compute
    tensorDescriptorEmbeddings.dataType = .float32
    tensorDescriptorEmbeddings.dimensions = .init([embeddingDimension, sequenceLength])!
    tensorDescriptorEmbeddings.strides = .init([1, embeddingDimension])!
    let embeddingsBuffer = device.makeBuffer(bytesNoCopy: &embeddingsData, length: MemoryLayout<Float>.stride * sequenceLength * embeddingDimension, options: .storageModeShared)!
    let embeddings = try! embeddingsBuffer.makeTensor(descriptor: tensorDescriptorEmbeddings, offset: 0)
    
    // Q, K, V weight tensors
    let tensorDescriptorWeights = MTLTensorDescriptor()
    tensorDescriptorWeights.usage = .compute
    tensorDescriptorWeights.dataType = .float32
    tensorDescriptorWeights.dimensions = .init([embeddingDimension, embeddingDimension])!
    tensorDescriptorWeights.strides = .init([1, embeddingDimension])!
    
    // Q weights
    let queryWeightsBuffer = device.makeBuffer(bytesNoCopy: &queryWeightData, length: MemoryLayout<Float>.stride * embeddingDimension * embeddingDimension, options: .storageModeShared)!
    let queryWeights = try! queryWeightsBuffer.makeTensor(descriptor: tensorDescriptorWeights, offset: 0)
    
    // K weights
    let keyWeightsBuffer = device.makeBuffer(bytesNoCopy: &keyWeightData, length: MemoryLayout<Float>.stride * embeddingDimension * embeddingDimension, options: .storageModeShared)!
    let keyWeights = try! keyWeightsBuffer.makeTensor(descriptor: tensorDescriptorWeights, offset: 0)
    
    // V weights
    let valueWeightsBuffer = device.makeBuffer(bytesNoCopy: &valueWeightData, length: MemoryLayout<Float>.stride * embeddingDimension * embeddingDimension, options: .storageModeShared)!
    let valueWeights = try! valueWeightsBuffer.makeTensor(descriptor: tensorDescriptorWeights, offset: 0)
    
    // Q, K, V tensors
    let tensorDescriptorQKV = MTLTensorDescriptor()
    tensorDescriptorQKV.usage = .compute
    tensorDescriptorQKV.dataType = .float32
    tensorDescriptorQKV.dimensions = .init([embeddingDimension, sequenceLength])!
    tensorDescriptorQKV.strides = .init([1, embeddingDimension])!
    
    // Q
    let queriesBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride * sequenceLength * embeddingDimension, options: .storageModeShared)!
    let queries = try! queriesBuffer.makeTensor(descriptor: tensorDescriptorQKV, offset: 0)
    
    // K
    let keysBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride * sequenceLength * embeddingDimension, options: .storageModeShared)!
    let keys = try! keysBuffer.makeTensor(descriptor: tensorDescriptorQKV, offset: 0)
    
    // V
    let valuesBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride * sequenceLength * embeddingDimension, options: .storageModeShared)!
    let values = try! valuesBuffer.makeTensor(descriptor: tensorDescriptorQKV, offset: 0)
    
    var sequenceLengthU32: UInt32 = UInt32(sequenceLength)
    let sequenceLengthBuffer = device.makeBuffer(bytesNoCopy: &sequenceLengthU32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    var headDimensionU32: UInt32 = UInt32(headDimension)
    let headDimensionBuffer = device.makeBuffer(bytesNoCopy: &headDimensionU32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    var headCountU32: UInt32 = UInt32(headCount)
    let headCountBuffer = device.makeBuffer(bytesNoCopy: &headCountU32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    
    var attentionScale = 1.0 / sqrt(Float(headDimension))
    let attentionScaleBuffer = device.makeBuffer(bytesNoCopy: &attentionScale, length: MemoryLayout<Float>.stride, options: .storageModeShared)!
    
    let commandBuffer = device.makeCommandBuffer()!
    let commandAllocator = device.makeCommandAllocator()!
    commandBuffer.beginCommandBuffer(allocator: commandAllocator, options: .init())
    
    // calculate Q, K, V
    let library = try! device.makeDefaultLibrary(bundle: .main)
    let kernelFunctionMatrixMultiply = library.makeFunction(name: "matrixMultiplyTensor")!
    let computePipelineStateMatrixMultiply = try! await device.makeComputePipelineState(function: kernelFunctionMatrixMultiply)
    let computeCommandEncoderQKV = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoderQKV.setComputePipelineState(computePipelineStateMatrixMultiply)
    
    // embeddings * queries
    let argumentTableDescriptorMatrixMultiply = MTL4ArgumentTableDescriptor()
    argumentTableDescriptorMatrixMultiply.maxBufferBindCount = 3
    let argumentTableQueryMatrixMultiply = try! device.makeArgumentTable(descriptor: argumentTableDescriptorMatrixMultiply)
    argumentTableQueryMatrixMultiply.setResource(embeddings.gpuResourceID, bufferIndex: 0)
    argumentTableQueryMatrixMultiply.setResource(queryWeights.gpuResourceID, bufferIndex: 1)
    argumentTableQueryMatrixMultiply.setResource(queries.gpuResourceID, bufferIndex: 2)
    computeCommandEncoderQKV.setArgumentTable(argumentTableQueryMatrixMultiply)
    let M = embeddings.dimensions.extents[1]
    let N = queryWeights.dimensions.extents[0]
    computeCommandEncoderQKV.dispatchThreadgroups(
        threadgroupsPerGrid: MTLSizeMake((N + 31)/32, (M + 63)/64, 1),
        threadsPerThreadgroup: MTLSizeMake(computePipelineStateMatrixMultiply.threadExecutionWidth * 4, 1, 1))
    
    // embeddings * keys
    let argumentTableKeysMatrixMultiply = try! device.makeArgumentTable(descriptor: argumentTableDescriptorMatrixMultiply)
    argumentTableKeysMatrixMultiply.setResource(embeddings.gpuResourceID, bufferIndex: 0)
    argumentTableKeysMatrixMultiply.setResource(keyWeights.gpuResourceID, bufferIndex: 1)
    argumentTableKeysMatrixMultiply.setResource(keys.gpuResourceID, bufferIndex: 2)
    computeCommandEncoderQKV.setArgumentTable(argumentTableKeysMatrixMultiply)
    computeCommandEncoderQKV.dispatchThreadgroups(
        threadgroupsPerGrid: MTLSizeMake((N + 31)/32, (M + 63)/64, 1),
        threadsPerThreadgroup: MTLSizeMake(computePipelineStateMatrixMultiply.threadExecutionWidth * 4, 1, 1))
    
    // embeddings * values
    let argumentTableValuesMatrixMultiply = try! device.makeArgumentTable(descriptor: argumentTableDescriptorMatrixMultiply)
    argumentTableValuesMatrixMultiply.setResource(embeddings.gpuResourceID, bufferIndex: 0)
    argumentTableValuesMatrixMultiply.setResource(valueWeights.gpuResourceID, bufferIndex: 1)
    argumentTableValuesMatrixMultiply.setResource(values.gpuResourceID, bufferIndex: 2)
    computeCommandEncoderQKV.setArgumentTable(argumentTableValuesMatrixMultiply)
    computeCommandEncoderQKV.dispatchThreadgroups(
        threadgroupsPerGrid: MTLSizeMake((N + 31)/32, (M + 63)/64, 1),
        threadsPerThreadgroup: MTLSizeMake(computePipelineStateMatrixMultiply.threadExecutionWidth * 4, 1, 1))
    computeCommandEncoderQKV.endEncoding()
    
    // reshape and transpose Q, K, V from (sequenceLength, embeddingDimension) to (headCount, sequenceLength, headDimension)
    // (1) create Q, K, and V reshaped tensors
    let tensorDescriptorQKVTransposed = MTLTensorDescriptor()
    tensorDescriptorQKVTransposed.usage = .compute
    tensorDescriptorQKVTransposed.dataType = .float32
    tensorDescriptorQKVTransposed.dimensions = .init([headDimension, sequenceLength, headCount])!
    tensorDescriptorQKVTransposed.strides = .init([1, headDimension, headDimension * sequenceLength])!
    
    let qBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride * headCount * sequenceLength * headDimension, options: .storageModeShared)!
    let q = try! qBuffer.makeTensor(descriptor: tensorDescriptorQKVTransposed, offset: 0)
    let kBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride * headCount * sequenceLength * headDimension, options: .storageModeShared)!
    let k = try! kBuffer.makeTensor(descriptor: tensorDescriptorQKVTransposed, offset: 0)
    let vBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride * headCount * sequenceLength * headDimension, options: .storageModeShared)!
    let v = try! vBuffer.makeTensor(descriptor: tensorDescriptorQKVTransposed, offset: 0)
    
    let kernelFunctionReshapeAndTransposeQKV = library.makeFunction(name: "reshapeAndTransposeQKVTensor")!
    let computePipelineStateReshapeAndTransposeQKV = try! await device.makeComputePipelineState(function: kernelFunctionReshapeAndTransposeQKV)
    let computeCommandEncoderReshapeAndTransposeQKV = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoderReshapeAndTransposeQKV.setComputePipelineState(computePipelineStateReshapeAndTransposeQKV)
    
    let argumentTableDescriptorReshapeAndTransposeQKV = MTL4ArgumentTableDescriptor()
    argumentTableDescriptorReshapeAndTransposeQKV.maxBufferBindCount = 5

    // (2) reshape and transpose Q
    let argumentTableReshapeAndTransposeQ = try! device.makeArgumentTable(descriptor: argumentTableDescriptorReshapeAndTransposeQKV)
    argumentTableReshapeAndTransposeQ.setResource(queries.gpuResourceID, bufferIndex: 0)
    argumentTableReshapeAndTransposeQ.setResource(q.gpuResourceID, bufferIndex: 1)
    argumentTableReshapeAndTransposeQ.setAddress(sequenceLengthBuffer.gpuAddress, index: 2)
    argumentTableReshapeAndTransposeQ.setAddress(headCountBuffer.gpuAddress, index: 3)
    argumentTableReshapeAndTransposeQ.setAddress(headDimensionBuffer.gpuAddress, index: 4)
    computeCommandEncoderReshapeAndTransposeQKV.setArgumentTable(argumentTableReshapeAndTransposeQ)
    
    // (3) a consumer queue barrier that blocks dispatch on Q, K, V creation
    computeCommandEncoderReshapeAndTransposeQKV.barrier(afterQueueStages: .dispatch, beforeStages: .dispatch)
    
    computeCommandEncoderReshapeAndTransposeQKV.dispatchThreads(
        threadsPerGrid: MTLSizeMake(sequenceLength, headCount, headDimension),
        threadsPerThreadgroup: MTLSizeMake(256, 1, 1))
    
    // (5) reshape and transpose K
    let argumentTableReshapeAndTransposeK = try! device.makeArgumentTable(descriptor: argumentTableDescriptorReshapeAndTransposeQKV)
    argumentTableReshapeAndTransposeK.setResource(keys.gpuResourceID, bufferIndex: 0)
    argumentTableReshapeAndTransposeK.setResource(k.gpuResourceID, bufferIndex: 1)
    argumentTableReshapeAndTransposeK.setAddress(sequenceLengthBuffer.gpuAddress, index: 2)
    argumentTableReshapeAndTransposeK.setAddress(headCountBuffer.gpuAddress, index: 3)
    argumentTableReshapeAndTransposeK.setAddress(headDimensionBuffer.gpuAddress, index: 4)
    computeCommandEncoderReshapeAndTransposeQKV.setArgumentTable(argumentTableReshapeAndTransposeK)
    computeCommandEncoderReshapeAndTransposeQKV.dispatchThreads(
        threadsPerGrid: MTLSizeMake(sequenceLength, headCount, headDimension),
        threadsPerThreadgroup: MTLSizeMake(256, 1, 1))
    
    // (6) reshape and transpose V
    let argumentTableReshapeAndTransposeV = try! device.makeArgumentTable(descriptor: argumentTableDescriptorReshapeAndTransposeQKV)
    argumentTableReshapeAndTransposeV.setResource(values.gpuResourceID, bufferIndex: 0)
    argumentTableReshapeAndTransposeV.setResource(v.gpuResourceID, bufferIndex: 1)
    argumentTableReshapeAndTransposeV.setAddress(sequenceLengthBuffer.gpuAddress, index: 2)
    argumentTableReshapeAndTransposeV.setAddress(headCountBuffer.gpuAddress, index: 3)
    argumentTableReshapeAndTransposeV.setAddress(headDimensionBuffer.gpuAddress, index: 4)
    computeCommandEncoderReshapeAndTransposeQKV.setArgumentTable(argumentTableReshapeAndTransposeV)
    computeCommandEncoderReshapeAndTransposeQKV.dispatchThreads(
        threadsPerGrid: MTLSizeMake(sequenceLength, headCount, headDimension),
        threadsPerThreadgroup: MTLSizeMake(256, 1, 1))
    
    computeCommandEncoderReshapeAndTransposeQKV.endEncoding()
    
    // attention scores S
    // attention scores tensor
    let tensorDescriptorAttentionScores = MTLTensorDescriptor()
    tensorDescriptorAttentionScores.usage = .compute
    tensorDescriptorAttentionScores.dataType = .float32
    tensorDescriptorAttentionScores.dimensions = .init([sequenceLength, sequenceLength, headCount])!
    tensorDescriptorAttentionScores.strides = .init([1, sequenceLength, sequenceLength * sequenceLength])!
    let attentionScoresBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride * headCount * sequenceLength * sequenceLength, options: .storageModeShared)!
    let attentionScores = try! attentionScoresBuffer.makeTensor(descriptor: tensorDescriptorAttentionScores, offset: 0)
    
    // (1) multiHeadScoresTensor is tensor version of multiHeadScore
    let kernelFunctionMultiHeadScores = library.makeFunction(name: "multiHeadScoresTensor")!
    let computePipelineStateMultiHeadScores = try! await device.makeComputePipelineState(function: kernelFunctionMultiHeadScores)
    let computeCommandEncoderMultiHeadScores = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoderMultiHeadScores.setComputePipelineState(computePipelineStateMultiHeadScores)
    
    let argumentTableDescriptorMultiHeadScores = MTL4ArgumentTableDescriptor()
    argumentTableDescriptorMultiHeadScores.maxBufferBindCount = 5
    let argumentTableMultiHeadScores = try! device.makeArgumentTable(descriptor: argumentTableDescriptorMultiHeadScores)
    argumentTableMultiHeadScores.setResource(q.gpuResourceID, bufferIndex: 0)
    argumentTableMultiHeadScores.setResource(k.gpuResourceID, bufferIndex: 1)
    argumentTableMultiHeadScores.setResource(attentionScores.gpuResourceID, bufferIndex: 2)
    argumentTableMultiHeadScores.setAddress(headDimensionBuffer.gpuAddress, index: 3)
    argumentTableMultiHeadScores.setAddress(attentionScaleBuffer.gpuAddress, index: 4)
    computeCommandEncoderMultiHeadScores.setArgumentTable(argumentTableMultiHeadScores)
    
    // (2) a consumer queue barrier that blocks dispatch on Q, K, V reshape and transpose
    computeCommandEncoderMultiHeadScores.barrier(afterQueueStages: .dispatch, beforeStages: .dispatch)
    
    computeCommandEncoderMultiHeadScores.dispatchThreads(
        threadsPerGrid: MTLSizeMake(sequenceLength, sequenceLength, headCount),
        threadsPerThreadgroup: MTLSizeMake(16, 16, 1))
    computeCommandEncoderMultiHeadScores.endEncoding()
    
    // softmax
    // attention weights tensor
    let tensorDescriptorAttentionWeights = MTLTensorDescriptor()
    tensorDescriptorAttentionWeights.usage = .compute
    tensorDescriptorAttentionWeights.dataType = .float32
    tensorDescriptorAttentionWeights.dimensions = .init([sequenceLength, sequenceLength, headCount])!
    tensorDescriptorAttentionWeights.strides = .init([1, sequenceLength, sequenceLength * sequenceLength])!
    let attentionWeightsBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride * headCount * sequenceLength * sequenceLength, options: .storageModeShared)!
    let attentionWeights = try! attentionWeightsBuffer.makeTensor(descriptor: tensorDescriptorAttentionWeights, offset: 0)
    
    // (1) softmax3DTensor is the tensor version of softmax3D
    let kernelFunctionSoftmax3D = library.makeFunction(name: "softmax3DTensor")!
    let computePipelineStateSoftmax3D = try! await device.makeComputePipelineState(function: kernelFunctionSoftmax3D)
    let computeCommandEncoderSoftmax3D = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoderSoftmax3D.setComputePipelineState(computePipelineStateSoftmax3D)
    
    let argumentTableDescriptorSoftmax3D = MTL4ArgumentTableDescriptor()
    argumentTableDescriptorSoftmax3D.maxBufferBindCount = 3
    let argumentTableSoftmax3D = try! device.makeArgumentTable(descriptor: argumentTableDescriptorSoftmax3D)
    argumentTableSoftmax3D.setResource(attentionScores.gpuResourceID, bufferIndex: 0)
    argumentTableSoftmax3D.setResource(attentionWeights.gpuResourceID, bufferIndex: 1)
    argumentTableSoftmax3D.setAddress(sequenceLengthBuffer.gpuAddress, index: 2)
    computeCommandEncoderSoftmax3D.setArgumentTable(argumentTableSoftmax3D)
    
    // (2) a consumer queue barrier that blocks dispatch on attention scores calculation
    computeCommandEncoderSoftmax3D.barrier(afterQueueStages: .dispatch, beforeStages: .dispatch)
    
    computeCommandEncoderSoftmax3D.dispatchThreads(
        threadsPerGrid: MTLSizeMake(32, sequenceLength, headCount),
        threadsPerThreadgroup: MTLSizeMake(32, 1, 1))
    computeCommandEncoderSoftmax3D.endEncoding()
    
    // context vectors
    // context vectors tensor
    let tensorDescriptorContextVectors = MTLTensorDescriptor()
    tensorDescriptorContextVectors.usage = .compute
    tensorDescriptorContextVectors.dataType = .float32
    tensorDescriptorContextVectors.dimensions = .init([headDimension, sequenceLength, headCount])!
    tensorDescriptorContextVectors.strides = .init([1, headDimension, headDimension * sequenceLength])!
    let contextVectorsBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride * headCount * sequenceLength * headDimension, options: .storageModeShared)!
    let contextVectors = try! contextVectorsBuffer.makeTensor(descriptor: tensorDescriptorContextVectors, offset: 0)
    
    // (1) multiHeadContextVectorsTensor is the tensor version of multiHeadContextVectors
    let kernelFunctionContextVectors = library.makeFunction(name: "multiHeadContextVectorsTensor")!
    let computePipelineStateContextVectors = try! await device.makeComputePipelineState(function: kernelFunctionContextVectors)
    let computeCommandEncoderContextVectors = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoderContextVectors.setComputePipelineState(computePipelineStateContextVectors)
    
    let argumentTableDescriptorContextVectors = MTL4ArgumentTableDescriptor()
    argumentTableDescriptorContextVectors.maxBufferBindCount = 5
    let argumentTableContextVectors = try! device.makeArgumentTable(descriptor: argumentTableDescriptorContextVectors)
    argumentTableContextVectors.setResource(attentionWeights.gpuResourceID, bufferIndex: 0)
    argumentTableContextVectors.setResource(v.gpuResourceID, bufferIndex: 1)
    argumentTableContextVectors.setResource(contextVectors.gpuResourceID, bufferIndex: 2)
    argumentTableContextVectors.setAddress(sequenceLengthBuffer.gpuAddress, index: 3)
    argumentTableContextVectors.setAddress(headDimensionBuffer.gpuAddress, index: 4)
    computeCommandEncoderContextVectors.setArgumentTable(argumentTableContextVectors)
    
    // (2) a consumer queue barrier that blocks dispatch on softmax calculation
    computeCommandEncoderContextVectors.barrier(afterQueueStages: .dispatch, beforeStages: .dispatch)
    
    computeCommandEncoderContextVectors.dispatchThreads(
        threadsPerGrid: MTLSizeMake(headDimension, sequenceLength, headCount),
        threadsPerThreadgroup: MTLSizeMake(8, 8, 8))
    computeCommandEncoderContextVectors.endEncoding()
    
    // transpose and flatten context vectors
    // context vectors reshaped and flattened tensor
    let tensorDescriptorContextVectorsFlattened = MTLTensorDescriptor()
    tensorDescriptorContextVectorsFlattened.usage = .compute
    tensorDescriptorContextVectorsFlattened.dataType = .float32
    tensorDescriptorContextVectorsFlattened.dimensions = .init([embeddingDimension, sequenceLength])!
    tensorDescriptorContextVectorsFlattened.strides = .init([1, embeddingDimension])!
    let contextVectorsFlattenedBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride * sequenceLength * embeddingDimension, options: .storageModeShared)!
    let contextVectorsFlattened = try! contextVectorsFlattenedBuffer.makeTensor(descriptor: tensorDescriptorContextVectorsFlattened, offset: 0)
    
    // (1) transposeAndFlattenContextVectorsTensor is the tensor version of transposeAndFlattenContextVectors
    let kernelFunctionTransposeAndFlattenContextVectors = library.makeFunction(name: "transposeAndFlattenContextVectorsTensor")!
    let computePipelineTransposeAndFlattenContextVectors = try! await device.makeComputePipelineState(function: kernelFunctionTransposeAndFlattenContextVectors)
    let computeCommandEncoderTransposeAndFlattenContextVectors = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoderTransposeAndFlattenContextVectors.setComputePipelineState(computePipelineTransposeAndFlattenContextVectors)
    
    let argumentTableDescriptorTransposeAndFlattenContextVectors = MTL4ArgumentTableDescriptor()
    argumentTableDescriptorTransposeAndFlattenContextVectors.maxBufferBindCount = 5
    let argumentTableTransposeAndFlattenContextVectors = try! device.makeArgumentTable(descriptor: argumentTableDescriptorTransposeAndFlattenContextVectors)
    argumentTableTransposeAndFlattenContextVectors.setResource(contextVectors.gpuResourceID, bufferIndex: 0)
    argumentTableTransposeAndFlattenContextVectors.setResource(contextVectorsFlattened.gpuResourceID, bufferIndex: 1)
    argumentTableTransposeAndFlattenContextVectors.setAddress(sequenceLengthBuffer.gpuAddress, index: 2)
    argumentTableTransposeAndFlattenContextVectors.setAddress(headCountBuffer.gpuAddress, index: 3)
    argumentTableTransposeAndFlattenContextVectors.setAddress(headDimensionBuffer.gpuAddress, index: 4)
    computeCommandEncoderTransposeAndFlattenContextVectors.setArgumentTable(argumentTableTransposeAndFlattenContextVectors)
    
    // (2) a consumer queue barrier that blocks dispatch on context vectors creation
    computeCommandEncoderTransposeAndFlattenContextVectors.barrier(afterQueueStages: .dispatch, beforeStages: .dispatch)
    
    computeCommandEncoderTransposeAndFlattenContextVectors.dispatchThreads(
        threadsPerGrid: MTLSizeMake(sequenceLength, headCount, headDimension),
        threadsPerThreadgroup: MTLSizeMake(8, 8, 8))
    computeCommandEncoderTransposeAndFlattenContextVectors.endEncoding()
    
    commandBuffer.endCommandBuffer()
    
    let residencySet = try! device.makeResidencySet(descriptor: .init())
    residencySet.addAllocation(embeddings)
    residencySet.addAllocation(queryWeights)
    residencySet.addAllocation(queries)
    residencySet.addAllocation(keyWeights)
    residencySet.addAllocation(keys)
    residencySet.addAllocation(valueWeights)
    residencySet.addAllocation(values)
    residencySet.addAllocation(sequenceLengthBuffer)
    residencySet.addAllocation(headDimensionBuffer)
    residencySet.addAllocation(q)
    residencySet.addAllocation(k)
    residencySet.addAllocation(v)
    residencySet.addAllocation(attentionScores)
    residencySet.addAllocation(attentionScaleBuffer)
    residencySet.addAllocation(attentionWeights)
    residencySet.addAllocation(contextVectors)
    residencySet.addAllocation(contextVectorsFlattened)
    residencySet.commit()
    
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
            
            print("multiHeadAttentionTensor GPU time: \(feedback.gpuEndTime - feedback.gpuStartTime) seconds\n")
        }
        commandQueue.commit([commandBuffer], options: commitOptions)
    }
    
    print(String(format: "multiHeadAttentionTensor %.4f seconds", CFAbsoluteTimeGetCurrent() - startTime))
    
    let ptr = contextVectorsFlattened.buffer!.contents().bindMemory(to: Float32.self, capacity: sequenceLength * embeddingDimension)
    let data = Array(UnsafeBufferPointer(start: ptr, count: sequenceLength * embeddingDimension))
    return Matrix(rows: sequenceLength, cols: embeddingDimension, data: data)
}

func multiHeadAttentionTensor() async {
    // sequenceLength, embeddingDimension, headCount
    let dimensions: [(Int, Int, Int)] = [
        (1024, 768, 12),
        (4096, 768, 12),
        (8192, 768, 12),
        (1024, 4096, 32),
        (4096, 4096, 32),
        (8192, 4096, 32),
    ]
    
    for dimension in dimensions {
        let sequenceLength = dimension.0
        let embeddingDimension = dimension.1
        let headCount = dimension.2
        
        print("sequenceLength: \(sequenceLength), embeddingDimension: \(embeddingDimension), headCount: \(headCount)")
        
        var embeddingsData = (0..<sequenceLength * embeddingDimension).map { _ in
            Float.randomNormal()
        }
        var (queryWeightData, keyWeightData, valueWeightData) = initQueryKeyValueWeights(embeddingDimension: embeddingDimension)
        
        let _ = await multiHeadAttentionTensor(
            sequenceLength: sequenceLength,
            embeddingDimension: embeddingDimension,
            headCount: headCount,
            embeddingsData: &embeddingsData,
            queryWeightData: &queryWeightData,
            keyWeightData: &keyWeightData,
            valueWeightData: &valueWeightData,
        )
    }
}
