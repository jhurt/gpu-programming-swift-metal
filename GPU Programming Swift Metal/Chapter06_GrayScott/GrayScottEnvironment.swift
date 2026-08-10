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

class GrayScottEnvironment {
    // classic growing worms
    static let coral = GrayScottParams(
        F: 0.0545, K: 0.0620, Du: 1.0, Dv: 0.5, dt: 1.0
    )
    
    // cells split and divide constantly
    static let mitosis = GrayScottParams(
        F: 0.0367, K: 0.0640, Du: 1.0, Dv: 0.5, dt: 1.0
    )
    
    // stable maze-like walls
    static let fingerprints = GrayScottParams(
        F: 0.0290, K: 0.0570, Du: 1.0, Dv: 0.5, dt: 1.0
    )
    
    // tiny, localized blue dots that move
    static let solitons = GrayScottParams(
        F: 0.0300, K: 0.0620, Du: 1.0, Dv: 0.5, dt: 1.0
    )
    
    // moving worms that look like gliders
    static let uSkate = GrayScottParams(
        F: 0.0620, K: 0.0609, Du: 1.0, Dv: 0.5, dt: 1.0
    )
    
    // inverted spots mostly filled with V with empty holes
    static let holes = GrayScottParams(
        F: 0.0390, K: 0.0580, Du: 1.0, Dv: 0.5, dt: 1.0
    )
    
    // pulsating, unstable patterns
    static let chaos = GrayScottParams(
        F: 0.0260, K: 0.0610, Du: 1.0, Dv: 0.5, dt: 1.0
    )
    
    // thicker, shorter segments than coral
    static let worms = GrayScottParams(
        F: 0.0780, K: 0.0610, Du: 1.0, Dv: 0.5, dt: 1.0
    )
    
    public static func getEnvironmentsForGrid(totalEnvironments: Int, gridWidth: Int, gridHeight: Int) -> [UInt32] {
        var cellEnvironments = [UInt32](repeating: 0, count: gridWidth * gridHeight)
        let biomeScale: Float = 0.003
        var i = 0
        for y in 0..<gridHeight {
            for x in 0..<gridWidth {
                let fx = Float(x) * biomeScale
                let fy = Float(y) * biomeScale
                
                // base biome noise
                var biome = GrayScottNoise.gradientNoise(fx, fy)
                // second octave
                biome = biome * 0.7 + GrayScottNoise.gradientNoise(fx * 2, fy * 2) * 0.3
                
                // quantize into environments domains
                let environmentID = min(Int(biome * Float(totalEnvironments)), totalEnvironments - 1)
                cellEnvironments[i] = UInt32(environmentID)
                i += 1
            }
        }
        return cellEnvironments
    }
}
