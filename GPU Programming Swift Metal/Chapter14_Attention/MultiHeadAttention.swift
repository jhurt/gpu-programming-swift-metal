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

private func reshapeAndTransposeQKV(input: inout Matrix,
                                    sequenceLength: Int,
                                    headDimension: Int,
                                    headCount: Int) async -> Matrix {
    let processor = ParallelProcessor(log: true)
    
    let inputBuffer = processor.device.makeBuffer(bytesNoCopy: &input.data, length: MemoryLayout<Float>.stride * input.rows * input.cols, options: .storageModeShared)!
    let outputBuffer = processor.device.makeBuffer(length: MemoryLayout<Float>.stride * headCount * sequenceLength * headDimension, options: .storageModeShared)!
    
    var sequenceLengthU32: UInt32 = UInt32(sequenceLength)
    let sequenceLengthBuffer = processor.device.makeBuffer(bytesNoCopy: &sequenceLengthU32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    var headCountU32: UInt32 = UInt32(headCount)
    let headCountBuffer = processor.device.makeBuffer(bytesNoCopy: &headCountU32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    var headDimensionU32: UInt32 = UInt32(headDimension)
    let headDimensionBuffer = processor.device.makeBuffer(bytesNoCopy: &headDimensionU32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    
    await processor.process(kernelName: "reshapeAndTransposeQKV",
                            buffers: inputBuffer, outputBuffer, sequenceLengthBuffer, headCountBuffer, headDimensionBuffer,
                            options: ProcessOptions(
                                threadsPerGridMTLSize: MTLSizeMake(sequenceLength, headCount, headDimension),
                                threadsPerThreadgroup: 256))
    
    let ptr = outputBuffer.contents().bindMemory(to: Float.self, capacity: headCount * sequenceLength * headDimension)
    let data = Array(UnsafeBufferPointer(start: ptr, count: headCount * sequenceLength * headDimension))
    
    return Matrix(rows: headCount * sequenceLength, cols: headDimension, data: data)
}

private func multiHeadScores(queries: inout Matrix,
                             keys: inout Matrix,
                             sequenceLength: Int,
                             headDimension: Int,
                             headCount: Int,
                             attentionScale: inout Float) async -> Matrix {
    let processor = ParallelProcessor(log: true)
    
    let queriesBuffer = processor.device.makeBuffer(bytesNoCopy: &queries.data, length: MemoryLayout<Float>.stride * queries.rows * queries.cols, options: .storageModeShared)!
    let keysBuffer = processor.device.makeBuffer(bytesNoCopy: &keys.data, length: MemoryLayout<Float>.stride * keys.rows * keys.cols, options: .storageModeShared)!
    let attentionScoresBuffer = processor.device.makeBuffer(length: MemoryLayout<Float>.stride * headCount * sequenceLength * sequenceLength, options: .storageModeShared)!
    
    var sequenceLengthU32: UInt32 = UInt32(sequenceLength)
    let sequenceLengthBuffer = processor.device.makeBuffer(bytesNoCopy: &sequenceLengthU32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    var headDimensionU32: UInt32 = UInt32(headDimension)
    let headDimensionBuffer = processor.device.makeBuffer(bytesNoCopy: &headDimensionU32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    let attentionScaleBuffer = processor.device.makeBuffer(bytesNoCopy: &attentionScale, length: MemoryLayout<Float>.stride, options: .storageModeShared)!
    
    await processor.process(kernelName: "multiHeadScores",
                            buffers: queriesBuffer, keysBuffer, attentionScoresBuffer, sequenceLengthBuffer, headDimensionBuffer, attentionScaleBuffer,
                            options: ProcessOptions(
                                threadsPerGridMTLSize: MTLSizeMake(sequenceLength, sequenceLength, headCount),
                                threadsPerThreadgroupMTLSize: MTLSizeMake(16, 16, 1)))
    
    let ptr = attentionScoresBuffer.contents().bindMemory(to: Float.self, capacity: headCount * sequenceLength * sequenceLength)
    let data = Array(UnsafeBufferPointer(start: ptr, count: headCount * sequenceLength * sequenceLength))
    return Matrix(rows: headCount * sequenceLength, cols: sequenceLength, data: data)
}

private func softmax3D(attentionScores: inout Matrix, sequenceLength: Int, headCount: Int) async -> Matrix {
    let processor = ParallelProcessor(log: true)
    
    let attentionScoresBuffer = processor.device.makeBuffer(bytesNoCopy: &attentionScores.data, length: MemoryLayout<Float>.stride * attentionScores.rows * attentionScores.cols, options: .storageModeShared)!
    let attentionWeightsBuffer = processor.device.makeBuffer(length: MemoryLayout<Float>.stride * attentionScores.rows * attentionScores.cols, options: .storageModeShared)!
    var sequenceLengthU32: UInt32 = UInt32(sequenceLength)
    let sequenceLengthBuffer = processor.device.makeBuffer(bytesNoCopy: &sequenceLengthU32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    
    await processor.process(kernelName: "softmax3D",
                            buffers: attentionScoresBuffer, attentionWeightsBuffer, sequenceLengthBuffer,
                            options: ProcessOptions(
                                threadsPerGridMTLSize: MTLSizeMake(32, sequenceLength, headCount),
                                threadsPerThreadgroup: 32))
    
    let ptr = attentionWeightsBuffer.contents().bindMemory(to: Float.self, capacity: attentionScores.rows * attentionScores.cols)
    let data = Array(UnsafeBufferPointer(start: ptr, count: attentionScores.rows * attentionScores.cols))
    return Matrix(rows: attentionScores.rows, cols: attentionScores.cols, data: data)
}

private func multiHeadContextVectors(attentionWeights: inout Matrix,
                                     values: inout Matrix,
                                     sequenceLength: Int,
                                     embeddingDimension: Int,
                                     headDimension: Int,
                                     headCount: Int,
                                     startTime: CFTimeInterval) async -> Matrix {
    let processor = ParallelProcessor(log: true)
    
    let attentionWeightsBuffer = processor.device.makeBuffer(bytesNoCopy: &attentionWeights.data, length: MemoryLayout<Float>.stride * attentionWeights.rows * attentionWeights.cols, options: .storageModeShared)!
    let valuesBuffer = processor.device.makeBuffer(bytesNoCopy: &values.data, length: MemoryLayout<Float>.stride * values.rows * values.cols, options: .storageModeShared)!
    let contextVectorsBuffer = processor.device.makeBuffer(length: MemoryLayout<Float>.stride * headCount * sequenceLength * headDimension, options: .storageModeShared)!
    
    var sequenceLengthU32: UInt32 = UInt32(sequenceLength)
    let sequenceLengthBuffer = processor.device.makeBuffer(bytesNoCopy: &sequenceLengthU32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    var headDimensionU32: UInt32 = UInt32(headDimension)
    let headDimensionBuffer = processor.device.makeBuffer(bytesNoCopy: &headDimensionU32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    
    await processor.process(kernelName: "multiHeadContextVectors",
                            buffers: attentionWeightsBuffer, valuesBuffer, contextVectorsBuffer, sequenceLengthBuffer, headDimensionBuffer,
                            options: ProcessOptions(
                                threadsPerGridMTLSize: MTLSizeMake(headDimension, sequenceLength, headCount),
                                threadsPerThreadgroupMTLSize: MTLSizeMake(8, 8, 8)))
    
    let flattenedContextVectorsBuffer = processor.device.makeBuffer(length: MemoryLayout<Float>.stride * sequenceLength * embeddingDimension, options: .storageModeShared)!
    var headCountU32: UInt32 = UInt32(headCount)
    let headCountBuffer = processor.device.makeBuffer(bytesNoCopy: &headCountU32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    await processor.process(kernelName: "transposeAndFlattenContextVectors",
                            buffers: contextVectorsBuffer, flattenedContextVectorsBuffer, sequenceLengthBuffer, headCountBuffer, headDimensionBuffer,
                            options: ProcessOptions(
                                threadsPerGridMTLSize: MTLSizeMake(sequenceLength, headCount, headDimension),
                                threadsPerThreadgroupMTLSize: MTLSizeMake(8, 8, 8)))
    
    print(String(format: "multiHeadAttention %.4f seconds\n", CFAbsoluteTimeGetCurrent() - startTime))
    
    let ptr = flattenedContextVectorsBuffer.contents().bindMemory(to: Float.self, capacity: sequenceLength * embeddingDimension)
    let data = Array(UnsafeBufferPointer(start: ptr, count: sequenceLength * embeddingDimension))
    return Matrix(rows: sequenceLength, cols: embeddingDimension, data: data)
}

private func multiHeadAttention(sequenceLength:Int,
                                embeddingDimension: Int,
                                headCount: Int,
                                embeddingsData: inout [Float],
                                queryWeightData: inout [Float],
                                keyWeightData: inout [Float],
                                valueWeightData: inout [Float]) async -> Matrix {
    let startTime = CFAbsoluteTimeGetCurrent()
    
    let headDimension = embeddingDimension / headCount
    
    var embeddings = Matrix(rows: sequenceLength, cols: embeddingDimension, data: embeddingsData)
    
    var queryWeights = Matrix(rows: embeddingDimension, cols: embeddingDimension, data: queryWeightData)
    var keyWeights = Matrix(rows: embeddingDimension, cols: embeddingDimension, data: keyWeightData)
    var valueWeights = Matrix(rows: embeddingDimension, cols: embeddingDimension, data: valueWeightData)
    
    var queries = await matrixMultiply(A: &embeddings, B: &queryWeights)
    var keys = await matrixMultiply(A: &embeddings, B: &keyWeights)
    var values = await matrixMultiply(A: &embeddings, B: &valueWeights)
    
    var attentionScale = 1.0 / sqrt(Float(headDimension))
    
    // (1) reshape and transpose Q, K, V from (sequenceLength, embeddingDimension) to (headCount, sequenceLength, headDimension)
    queries = await reshapeAndTransposeQKV(
        input: &queries,
        sequenceLength: sequenceLength,
        headDimension: headDimension,
        headCount: headCount)
    keys = await reshapeAndTransposeQKV(
        input: &keys,
        sequenceLength: sequenceLength,
        headDimension: headDimension,
        headCount: headCount)
    values = await reshapeAndTransposeQKV(
        input: &values,
        sequenceLength: sequenceLength,
        headDimension: headDimension,
        headCount: headCount)
    
    // (2) calculate a single S matrix with scores from all heads, shaped (headCount, sequenceLength, sequenceLength)
    var attentionScores = await multiHeadScores(queries: &queries,
                                                keys: &keys,
                                                sequenceLength: sequenceLength,
                                                headDimension: headDimension,
                                                headCount: headCount,
                                                attentionScale: &attentionScale)
    
    // (3) use a 3D softmax kernel to calculate a single P matrix with probabilities from all heads, shaped (headCount, sequenceLength, sequenceLength)
    var attentionWeights = await softmax3D(attentionScores: &attentionScores, sequenceLength: sequenceLength, headCount: headCount)
    
    // (4) compute context vectors for all heads, shaped (sequenceLength, embeddingDimension)
    let contextVectors = await multiHeadContextVectors(attentionWeights: &attentionWeights,
                                                       values: &values,
                                                       sequenceLength: sequenceLength,
                                                       embeddingDimension: embeddingDimension,
                                                       headDimension: headDimension,
                                                       headCount: headCount,
                                                       startTime: startTime,
    )
    
    
    return contextVectors
}

private func multiHeadAttentionOnlineSoftmax(sequenceLength:Int,
                                             embeddingDimension: Int,
                                             headCount: Int,
                                             embeddingsData: inout [Float],
                                             queryWeightData: inout [Float],
                                             keyWeightData: inout [Float],
                                             valueWeightData: inout [Float]) async -> Matrix {
    let startTime = CFAbsoluteTimeGetCurrent()
    
    let processor = ParallelProcessor(log: true)
    let headDimension = embeddingDimension / headCount
    
    var embeddings = Matrix(rows: sequenceLength, cols: embeddingDimension, data: embeddingsData)
    
    var queryWeights = Matrix(rows: embeddingDimension, cols: embeddingDimension, data: queryWeightData)
    var keyWeights = Matrix(rows: embeddingDimension, cols: embeddingDimension, data: keyWeightData)
    var valueWeights = Matrix(rows: embeddingDimension, cols: embeddingDimension, data: valueWeightData)
    
    var queries = await matrixMultiply(A: &embeddings, B: &queryWeights)
    var keys = await matrixMultiply(A: &embeddings, B: &keyWeights)
    var values = await matrixMultiply(A: &embeddings, B: &valueWeights)
    
    var attentionScale = 1.0 / sqrt(Float(headDimension))
    let attentionScaleBuffer = processor.device.makeBuffer(bytesNoCopy: &attentionScale, length: MemoryLayout<Float>.stride, options: .storageModeShared)!
    
    let contextVectorsBuffer = processor.device.makeBuffer(length: MemoryLayout<Float>.stride * headCount * sequenceLength * headDimension, options: .storageModeShared)!
    
    var sequenceLengthU32: UInt32 = UInt32(sequenceLength)
    let sequenceLengthBuffer = processor.device.makeBuffer(bytesNoCopy: &sequenceLengthU32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    var headDimensionU32: UInt32 = UInt32(headDimension)
    let headDimensionBuffer = processor.device.makeBuffer(bytesNoCopy: &headDimensionU32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    var headCountU32: UInt32 = UInt32(headCount)
    let headCountBuffer = processor.device.makeBuffer(bytesNoCopy: &headCountU32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    
    // (1) reshape and transpose Q, K, V from (sequenceLength, embeddingDimension) to (headCount, sequenceLength, headDimension)
    async let q = await reshapeAndTransposeQKV(
        input: &queries,
        sequenceLength: sequenceLength,
        headDimension: headDimension,
        headCount: headCount)
    async let k = await reshapeAndTransposeQKV(
        input: &keys,
        sequenceLength: sequenceLength,
        headDimension: headDimension,
        headCount: headCount)
    async let v = await reshapeAndTransposeQKV(
        input: &values,
        sequenceLength: sequenceLength,
        headDimension: headDimension,
        headCount: headCount)
    (queries, keys, values) = await (q, k, v)
    
    let queriesBuffer = processor.device.makeBuffer(bytesNoCopy: &queries.data, length: MemoryLayout<Float>.stride * queries.rows * queries.cols, options: .storageModeShared)!
    let keysBuffer = processor.device.makeBuffer(bytesNoCopy: &keys.data, length: MemoryLayout<Float>.stride * keys.rows * keys.cols, options: .storageModeShared)!
    let valuesBuffer = processor.device.makeBuffer(bytesNoCopy: &values.data, length: MemoryLayout<Float>.stride * values.rows * values.cols, options: .storageModeShared)!
    
    await processor.process(kernelName: "multiHeadAttentionOnlineSoftmax",
                            buffers: queriesBuffer, keysBuffer, valuesBuffer, contextVectorsBuffer, sequenceLengthBuffer, headCountBuffer, headDimensionBuffer, attentionScaleBuffer,
                            options: ProcessOptions(
                                threadsPerGridMTLSize: MTLSizeMake(32, sequenceLength, headCount),
                                threadsPerThreadgroup: 32))
    
    let flattenedContextVectorsBuffer = processor.device.makeBuffer(length: MemoryLayout<Float>.stride * sequenceLength * embeddingDimension, options: .storageModeShared)!
    await processor.process(kernelName: "transposeAndFlattenContextVectors",
                            buffers: contextVectorsBuffer, flattenedContextVectorsBuffer, sequenceLengthBuffer, headCountBuffer, headDimensionBuffer,
                            options: ProcessOptions(
                                threadsPerGridMTLSize: MTLSizeMake(sequenceLength, headCount, headDimension),
                                threadsPerThreadgroupMTLSize: MTLSizeMake(8, 8, 8)))
    
    print(String(format: "multiHeadAttentionOnlineSoftmax %.4f seconds\n", CFAbsoluteTimeGetCurrent() - startTime))
    
    let ptr = flattenedContextVectorsBuffer.contents().bindMemory(to: Float.self, capacity: sequenceLength * embeddingDimension)
    let data = Array(UnsafeBufferPointer(start: ptr, count: sequenceLength * embeddingDimension))
    let result = Matrix(rows: sequenceLength, cols: embeddingDimension, data: data)
    
    return result
}

func multiHeadAttention() async {
    // sequenceLength, embeddingDimension, headCount
    let dimensions: [(Int, Int, Int)] = [
        (4608, 3072, 24),
        
        (1024, 768, 12),
        (4096, 768, 12),
        (8192, 768, 12),
        (16384, 768, 12),
        
        (1024, 4096, 32),
        (4096, 4096, 32),
        (8192, 4096, 32),
        (16384, 4096, 32),
        
        (1024, 8192, 64),
        (4096, 8192, 64),
        (8192, 8192, 64),
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
        
        let contextVectors = await multiHeadAttention(
            sequenceLength: sequenceLength,
            embeddingDimension: embeddingDimension,
            headCount: headCount,
            embeddingsData: &embeddingsData,
            queryWeightData: &queryWeightData,
            keyWeightData: &keyWeightData,
            valueWeightData: &valueWeightData
        )
        
        let contextVectorsOnlineSoftmax = await multiHeadAttentionOnlineSoftmax(
            sequenceLength: sequenceLength,
            embeddingDimension: embeddingDimension,
            headCount: headCount,
            embeddingsData: &embeddingsData,
            queryWeightData: &queryWeightData,
            keyWeightData: &keyWeightData,
            valueWeightData: &valueWeightData,
        )
        
        print("multiHeadAttention: \(contextVectors)")
        print("multiHeadAttentionOnlineSoftmax: \(contextVectorsOnlineSoftmax)")
        
        compareMatricesRanges(A: contextVectors, B: contextVectorsOnlineSoftmax)
    }
}
