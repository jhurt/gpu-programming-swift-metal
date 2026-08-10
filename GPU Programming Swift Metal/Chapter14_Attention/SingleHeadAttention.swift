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

private func softmax2D(attentionScores: inout Matrix, attentionScale: inout Float) async -> Matrix {
    let processor = ParallelProcessor(log: true)
    
    let attentionScoresBuffer = processor.device.makeBuffer(bytesNoCopy: &attentionScores.data, length: MemoryLayout<Float>.stride * attentionScores.rows * attentionScores.cols, options: .storageModeShared)!
    let attentionWeightsBuffer = processor.device.makeBuffer(length: MemoryLayout<Float>.stride * attentionScores.rows * attentionScores.cols, options: .storageModeShared)!
    var sequenceLength = UInt32(attentionScores.cols)
    let sequenceLengthBuffer = processor.device.makeBuffer(bytes: &sequenceLength, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    let attentionScaleBuffer = processor.device.makeBuffer(bytes: &attentionScale, length: MemoryLayout<Float>.stride, options: .storageModeShared)!
    
    await processor.process(kernelName: "softmax2D",
                            buffers: attentionScoresBuffer, attentionWeightsBuffer, sequenceLengthBuffer, attentionScaleBuffer,
                            options: ProcessOptions(threadgroupsPerGrid: attentionScores.rows, threadsPerThreadgroup: 32))
    
    let ptr = attentionWeightsBuffer.contents().bindMemory(to: Float.self, capacity: attentionScores.rows * attentionScores.cols)
    let data = Array(UnsafeBufferPointer(start: ptr, count: attentionScores.rows * attentionScores.cols))
    return Matrix(rows: attentionScores.rows, cols: attentionScores.cols, data: data)
}

private func readSavedWeights(embeddingDimension: Int, name: String) -> [Float] {
    let directory = FileManager.default.temporaryDirectory
    let targetUrl = directory.appendingPathComponent("\(name)-\(embeddingDimension).weights")
    if FileManager.default.fileExists(atPath: targetUrl.path) {
        print("loading \(targetUrl.absoluteString) from disk")
        return loadFloatArrayFromDisk(sourceUrl: targetUrl)
    } else {
        let xavierStdDev = sqrt(2.0 / Float(embeddingDimension + embeddingDimension))
        let clampBoundary = 3.0 * xavierStdDev // prevent overflow
        let weightData = (0..<embeddingDimension * embeddingDimension).map { _ in
            Float.randomNormal(mean: 0.0, standardDeviation: xavierStdDev, clampBoundary: clampBoundary)
        }
        print("saving \(targetUrl.absoluteString) to disk")
        saveFloatArrayToDisk(floatArray: weightData, targetUrl: targetUrl)
        return weightData
    }
}

func initQueryKeyValueWeights(embeddingDimension: Int) -> ([Float], [Float], [Float]) {
    let queryWeightData = readSavedWeights(embeddingDimension: embeddingDimension, name: "q")
    let keyWeightData = readSavedWeights(embeddingDimension: embeddingDimension, name: "k")
    let valueWeightData = readSavedWeights(embeddingDimension: embeddingDimension, name: "v")
    return (queryWeightData, keyWeightData, valueWeightData)
}

private func attention(embeddingsData: [Float]) async -> Matrix {
    // (1) number of tokens in the input sequence
    let sequenceLength = 4096
    // (2) the size of the token and positional embeddings, here we match GPT-2 small
    let embeddingDimension = 3072
    
    // (3) create an embedding matrix
    var embeddings = Matrix(rows: sequenceLength, cols: embeddingDimension, data: embeddingsData)
    
    // (4) initialize the Q, K, and V weight matrices, often abbreviated as Wq, Wk, and Wv in the literature
    let (queryWeightData, keyWeightData, valueWeightData) = initQueryKeyValueWeights(embeddingDimension: embeddingDimension)
    var queryWeights = Matrix(rows: embeddingDimension, cols: embeddingDimension, data: queryWeightData)
    var keyWeights = Matrix(rows: embeddingDimension, cols: embeddingDimension, data: keyWeightData)
    var valueWeights = Matrix(rows: embeddingDimension, cols: embeddingDimension, data: valueWeightData)
    
    // (5) realize Q, K, and V by multiplying the embedding matrix by Wq, Wk, and Wv, respectively
    var queries = await matrixMultiply(A: &embeddings, B: &queryWeights)
    var keys = await matrixMultiply(A: &embeddings, B: &keyWeights)
    var values = await matrixMultiply(A: &embeddings, B: &valueWeights)
    
    // (6) calculate attention scores S
    var keysTranspose = await matrixTranspose(&keys)
    var attentionScores = await matrixMultiply(A: &queries, B: &keysTranspose)
    
    // (7) calculate the probability matrix P
    var attentionScale = 1.0 / sqrt(Float(embeddingDimension))
    var attentionWeights = await softmax2D(attentionScores: &attentionScores, attentionScale: &attentionScale)
    
    // (8) multiply the attention weights by V to get the context vector matrix
    return await matrixMultiply(A: &attentionWeights, B: &values)
}

func singleHeadAttention() async {
    let sequenceLength = 4096
    // (2) the size of the token and positional embeddings, here we match GPT-2 small
    let embeddingDimension = 3072
    
    // (3) create an embedding matrix
    let embeddingsData = (0..<sequenceLength * embeddingDimension).map { _ in
        Float.randomNormal()
    }

    let startTime = CFAbsoluteTimeGetCurrent()
    let _ = await attention(embeddingsData: embeddingsData)
    print(String(format: "singleHeadAttention %.4f seconds", CFAbsoluteTimeGetCurrent() - startTime))
}
