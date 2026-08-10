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

class GrayScottNoise {
    private static func gradientHash(_ x: Int, _ y: Int) -> UInt32 {
        var h = UInt32(x) &* 374761393 &+ UInt32(y) &* 668265263
        h ^= h >> 13
        h &*= 1274126177
        return h
    }
    
    private static func gradient(_ ix: Int, _ iy: Int) -> (Float, Float) {
        let h = gradientHash(ix, iy)
        let angle = Float(h) * (2.0 * .pi / Float(UInt32.max))
        return (cos(angle), sin(angle))
    }
    
    public static func gradientNoise(_ x: Float, _ y: Float) -> Float {
        let ix = Int(floor(x))
        let iy = Int(floor(y))
        
        let fx = x - Float(ix)
        let fy = y - Float(iy)
        
        // smoothstep
        let ux = fx * fx * (3 - 2 * fx)
        let uy = fy * fy * (3 - 2 * fy)
        
        let g00 = gradient(ix,     iy)
        let g10 = gradient(ix + 1, iy)
        let g01 = gradient(ix,     iy + 1)
        let g11 = gradient(ix + 1, iy + 1)
        
        let n00 = g00.0 * fx       + g00.1 * fy
        let n10 = g10.0 * (fx - 1) + g10.1 * fy
        let n01 = g01.0 * fx       + g01.1 * (fy - 1)
        let n11 = g11.0 * (fx - 1) + g11.1 * (fy - 1)
        
        let nx0 = n00 + (n10 - n00) * ux
        let nx1 = n01 + (n11 - n01) * ux
        let nxy = nx0 + (nx1 - nx0) * uy
        
        // normalize to [0,1]
        return 0.5 + 0.5 * nxy
    }
    
    private static func valueNoiseHash(_ x: Int, _ y: Int) -> Float {
        var h = x &* 374761393 &+ y &* 668265263
        h = (h ^ (h >> 13)) &* 1274126177
        return Float(h & 0x7fffffff) / Float(Int32.max)
    }
    
    public static func valueNoise(_ x: Float, _ y: Float) -> Float {
        let ix = Int(floor(x))
        let iy = Int(floor(y))
        let fx = x - Float(ix)
        let fy = y - Float(iy)
        
        let a = valueNoiseHash(ix,     iy)
        let b = valueNoiseHash(ix + 1, iy)
        let c = valueNoiseHash(ix,     iy + 1)
        let d = valueNoiseHash(ix + 1, iy + 1)
        
        // smoothstep
        let ux = fx * fx * (3 - 2 * fx)
        let uy = fy * fy * (3 - 2 * fy)
        
        // interpolate x
        let ab = a + (b - a) * ux
        let cd = c + (d - c) * ux
        
        // interpolate y
        return ab + (cd - ab) * uy
    }
}
