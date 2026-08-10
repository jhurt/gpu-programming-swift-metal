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

import Combine
import SwiftUI
import CoreGraphics

@MainActor
final class MandelbrotStore: ObservableObject {
    @Published var image: CGImage?
    
    /// Accepts a single `[UInt8]` pixel buffer. Supports RGB24 or RGBA32.
    /// - If `pixels.count == width*height*3` → RGB24 (no alpha)
    /// - If `pixels.count == width*height*4` → RGBA32 (non-premultiplied alpha)
    func update(from pixels: [UInt8], width: Int, height: Int, premultiplied: Bool = false) {
        let pxCount = width * height
        guard pixels.count == pxCount * 3 || pixels.count == pxCount * 4 else {
            preconditionFailure("pixels.count must be width*height*3 (RGB) or *4 (RGBA)")
        }
        
        let channels = pixels.count / pxCount
        let cs = CGColorSpaceCreateDeviceRGB()
        let bitsPerComponent = 8
        let bytesPerRow = width * channels
        let bitsPerPixel = channels * 8
        
        // Choose bitmapInfo based on channels
        let bitmapInfo: CGBitmapInfo = {
            if channels == 3 {
                // 24-bit RGB, no alpha
                return CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
            } else {
                // 32-bit RGBA; default to non-premultiplied
                let alpha: CGImageAlphaInfo = premultiplied ? .premultipliedLast : .noneSkipLast
                return [.byteOrder32Big, CGBitmapInfo(rawValue: alpha.rawValue)]
            }
        }()
        
        guard let provider = CGDataProvider(data: Data(pixels) as CFData) else {
            self.image = nil
            return
        }
        
        self.image = CGImage(
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bitsPerPixel: bitsPerPixel,
            bytesPerRow: bytesPerRow,
            space: cs,
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        )
    }
}
