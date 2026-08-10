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

func saveFloatArrayToDisk(floatArray: [Float], targetUrl: URL) {
    let binaryData = floatArray.withUnsafeBytes { (bufferPointer: UnsafeRawBufferPointer) -> Data in
        return Data(bufferPointer)
    }
    
    try! binaryData.write(to: targetUrl, options: .atomic)
}

func loadFloatArrayFromDisk(sourceUrl: URL) -> [Float] {
    let binaryData = try! Data(contentsOf: sourceUrl)
    let elementCount = binaryData.count / MemoryLayout<Float>.size
    return binaryData.withUnsafeBytes { (bufferPointer: UnsafeRawBufferPointer) -> [Float] in
        let typedPointer = bufferPointer.bindMemory(to: Float.self)
        return Array(typedPointer.prefix(elementCount))
    }
}
