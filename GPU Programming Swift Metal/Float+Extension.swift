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

extension Float {
    static func randomNormal(mean: Float = 0.0, standardDeviation: Float = 1.0, clampBoundary: Float = 100.0) -> Float {
        let firstUnitRandom = Float.random(in: 0..<1)
        let secondUnitRandom = Float.random(in: 0..<1)
        let radialComponent = sqrt(-2.0 * log(firstUnitRandom))
        let angularComponent = cos(2.0 * Float.pi * secondUnitRandom)
        
        let val = mean + (radialComponent * angularComponent * standardDeviation)
        return max(-clampBoundary, min(clampBoundary, val))
    }
}
