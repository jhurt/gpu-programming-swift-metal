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

nonisolated private func sequential(width: Int, height: Int, redMap: [UInt8], greenMap: [UInt8], blueMap: [UInt8]) -> [UInt8] {
    // (1) zoom into a small part of the fractal
    let realMin:Float32 = -0.7801785714285
    let realMax:Float32 = -0.7676785714285
    let imaginaryMin:Float32 = -0.1279296875
    let imaginaryMax:Float32 = -0.1181640625
    
    let scaleReal = (realMax - realMin) / Float32(width)
    let scaleImaginary = (imaginaryMax - imaginaryMin) / Float32(height)
    
    // (2) iterate all (width * height) pixels
    var pixels = [UInt8](repeating: 0, count: width * height * 3)
    for i in 0..<width {
        for j in 0..<height {
            let cReal = realMin + (Float32(i) * scaleReal)
            let cImaginary = imaginaryMin + (Float32(j) * scaleImaginary)
            var zReal:Float32 = 0.0
            var zImaginary:Float32 = 0.0
            var colorMapIndex = 0
            while colorMapIndex < 255 {
                // (3) Z_n+1 = Z_n^2 + C
                let lengthSquared = (zReal * zReal) + (zImaginary * zImaginary)
                if lengthSquared >= 4.0 {
                    // Z has exceeded 2.0
                    break
                }
                
                let zNextReal = ((zReal * zReal) - (zImaginary * zImaginary)) + cReal
                let zNextImaginary = (2.0 * zReal * zImaginary) + cImaginary
                zReal = zNextReal
                zImaginary = zNextImaginary
                colorMapIndex += 1
            }
            
            // (4) look up the pixel color from the map
            let pixelIndex = (j * width + i) * 3
            pixels[pixelIndex] = redMap[colorMapIndex]
            pixels[pixelIndex + 1] = greenMap[colorMapIndex]
            pixels[pixelIndex + 2] = blueMap[colorMapIndex]
        }
    }
    
    return pixels
}

private func parallel(width: Int, height: Int, redMap: inout [UInt8], greenMap: inout [UInt8], blueMap: inout [UInt8]) async -> [UInt8] {
    let processor = ParallelProcessor()
    
    // (1) create RGB color map buffers
    let redMapBuffer = processor.device.makeBuffer(bytesNoCopy: &redMap, length: MemoryLayout<UInt8>.stride * redMap.count, options: .storageModeShared)!
    let greenMapBuffer = processor.device.makeBuffer(bytesNoCopy: &greenMap, length: MemoryLayout<UInt8>.stride * greenMap.count, options: .storageModeShared)!
    let blueMapBuffer = processor.device.makeBuffer(bytesNoCopy: &blueMap, length: MemoryLayout<UInt8>.stride * blueMap.count, options: .storageModeShared)!

    // (2) create `min` and `scale` SIMD2<Float> buffers
    let realMin = Float(-0.7801785714285)
    let imaginaryMin = Float(-0.1279296875);
    var min = SIMD2<Float>(realMin, imaginaryMin)
    let minBuffer = processor.device.makeBuffer(bytes: &min, length: MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared)!

    let realMax = Float(-0.7676785714285)
    let imaginaryMax = Float(-0.1181640625);

    let scaleReal = (realMax - realMin) / Float(width)
    let scaleImaginary = (imaginaryMax - imaginaryMin) / Float(height)
    var scale = SIMD2<Float>(scaleReal, scaleImaginary)
    let scaleBuffer = processor.device.makeBuffer(bytes: &scale, length: MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared)!
    
    // (3) specify image width and height as single item UInt32 buffers
    var widthU32 = UInt32(width)
    let widthBuffer = processor.device.makeBuffer(bytes: &widthU32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    var heightU32 = UInt32(height)
    let heightBuffer = processor.device.makeBuffer(bytes: &heightU32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    
    // (4) create an output pixel buffer of width * height * RGB
    let pixelCount = width * height * 3
    let pixelBuffer = processor.device.makeBuffer(length: MemoryLayout<UInt8>.stride * pixelCount, options: .storageModeShared)!
    
    // (5) launch a 2D grid of width * height threads
    let options = ProcessOptions(threadsPerGridMTLSize: MTLSizeMake(width, height, 1))
    await processor.process(
        kernelName: "mandelbrot",
        buffers: redMapBuffer, greenMapBuffer, blueMapBuffer, minBuffer, scaleBuffer, widthBuffer, heightBuffer, pixelBuffer,
        options: options
    )
    
    // (6) bind the output pixel memory to a Swift array
    let pixelPtr = pixelBuffer.contents().bindMemory(to: UInt8.self, capacity: pixelCount)
    return Array(UnsafeBufferPointer(start: pixelPtr, count: pixelCount))
}

nonisolated func mandelbrot(width: Int, height: Int) async -> [UInt8] {
    // randomly pick a color map
    let resourcesURL = Bundle.main.resourceURL!
    let mandelbrotFolderURL = resourcesURL.appendingPathComponent("mandelbrot", isDirectory: true)
    let fileURLs = try! FileManager.default.contentsOfDirectory(
        at: mandelbrotFolderURL,
        includingPropertiesForKeys: nil,
        options: .skipsHiddenFiles
    )
    let colorMapURLs = fileURLs.filter { $0.pathExtension == "map" }
    let colorMapURL = colorMapURLs.randomElement()!

    // parse the color map file contents into R G B arrays
    let text = try! String(contentsOf: colorMapURL, encoding: .utf8)
    let lines = text.split(whereSeparator: \.isNewline).map(String.init)
    var redMap = [UInt8]()
    var greenMap = [UInt8]()
    var blueMap = [UInt8]()
    for line in lines {
        let tokens = line.split(separator: " ")
        redMap.append(UInt8(tokens[0]) ?? 0)
        greenMap.append(UInt8(tokens[1]) ?? 0)
        blueMap.append(UInt8(tokens[2]) ?? 0)
    }
    
    var startTime = CFAbsoluteTimeGetCurrent()
    let parallelPixels = await parallel(width: width, height: height, redMap: &redMap, greenMap: &greenMap, blueMap: &blueMap)
    print(String(format: "Parallel elapsed time: %.4f seconds\n", CFAbsoluteTimeGetCurrent() - startTime))
    
    /*
    startTime = CFAbsoluteTimeGetCurrent()
    let _ = sequential(width: width, height: height, redMap: redMap, greenMap: greenMap, blueMap: blueMap)
    print(String(format: "Sequential elapsed time: %.4f seconds\n", CFAbsoluteTimeGetCurrent() - startTime))
    */
    return parallelPixels
}
