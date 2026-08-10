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

struct Matrix {
    let rows: Int
    let cols: Int
    public var data: [Float]
    
    init(rows: Int, cols: Int, repeating value: Float) {
        self.rows = rows
        self.cols = cols
        self.data = Array(repeating: value, count: rows * cols)
    }
    
    init(rows: Int, cols: Int, data: [Float]) {
        precondition(data.count == rows * cols, "Data count must equal rows × cols")
        self.rows = rows
        self.cols = cols
        self.data = data
    }

    subscript(row: Int, col: Int) -> Float {
        get {
            precondition(row >= 0 && row < rows && col >= 0 && col < cols, "Index out of bounds")
            return data[row * cols + col]
        }
        set {
            precondition(row >= 0 && row < rows && col >= 0 && col < cols, "Index out of bounds")
            data[row * cols + col] = newValue
        }
    }
    
    mutating func toMetalBuffer(device: MTLDevice) -> MTLBuffer {
        var metalBuffer: MTLBuffer!
        let count = data.count
        data.withUnsafeMutableBufferPointer { dataPtr in
            metalBuffer = device.makeBuffer(bytesNoCopy: dataPtr.baseAddress!, length: MemoryLayout<Float>.stride * count, options: .storageModeShared)!
        }
        return metalBuffer
    }
}

extension Matrix: CustomStringConvertible {
    var description: String {
        var output = "\(rows) x \(cols) matrix, \((MemoryLayout<Float>.stride * rows * cols) / 1_000_000) MB\n"
        for i in 0..<min(rows, 10) {
            var rowString = "["
            for j in 0..<min(cols, 10) {
                let index = i * cols + j
                rowString += "\(data[index])"
                if j < cols - 1 {
                    rowString += ", "
                }
            }
            rowString += "]\n"
            output += rowString
        }
        
        return output
    }
}

extension Matrix {
    mutating func toMPSMatrix(device: MTLDevice) -> MPSMatrix {
        let buffer = toMetalBuffer(device: device)
        let descriptor = MPSMatrixDescriptor(
            rows: rows,
            columns: cols,
            rowBytes: cols * MemoryLayout<Float>.stride,
            dataType: .float32
        )
        return MPSMatrix(buffer: buffer, descriptor: descriptor)
    }
}

extension Matrix {
    static func fromMPSMatrix(_ mpsMatrix: MPSMatrix, type: Float.Type) -> Matrix {
        let rows = mpsMatrix.rows
        let cols = mpsMatrix.columns
        let count = rows * cols
        
        let ptr = mpsMatrix.data.contents().bindMemory(to: Float.self, capacity: count)
        let data = Array(UnsafeBufferPointer(start: ptr, count: count))
        return Matrix(rows: rows, cols: cols, data: data)
    }
}
