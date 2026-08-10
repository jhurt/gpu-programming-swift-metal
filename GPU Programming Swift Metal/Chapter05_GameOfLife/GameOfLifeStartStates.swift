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

class GameOfLifeStartStates {
    static func createGosperGliderGunGrid(width: Int, height: Int) -> [UInt8] {
        var grid = [UInt8](repeating: 0, count: width * height)
        
        // Calculate offsets to center the gun
        // The pattern is roughly 36x9, so we shift it left/up by half that size
        let offsetX = (width / 2) - 18
        let offsetY = (height / 2) - 5
        
        // Helper function to safely set cells
        func setCell(_ x: Int, _ y: Int) {
            let finalX = offsetX + x
            let finalY = offsetY + y
            
            if finalX >= 0 && finalX < width && finalY >= 0 && finalY < height {
                grid[finalY * width + finalX] = 1
            }
        }
        
        // --- Gosper's Glider Gun Coordinates ---
        
        // 1. Left Square Block
        setCell(1, 5); setCell(2, 5)
        setCell(1, 6); setCell(2, 6)
        
        // 2. Left "Hive" (The main mechanism)
        setCell(11, 5); setCell(11, 6); setCell(11, 7)
        setCell(12, 4); setCell(12, 8)
        setCell(13, 3); setCell(13, 9)
        setCell(14, 3); setCell(14, 9)
        setCell(15, 6) // Center dot
        setCell(16, 4); setCell(16, 8)
        setCell(17, 5); setCell(17, 6); setCell(17, 7)
        setCell(18, 6) // Connector
        
        // 3. Right "Mouth" (The ejector)
        setCell(21, 3); setCell(21, 4); setCell(21, 5)
        setCell(22, 3); setCell(22, 4); setCell(22, 5)
        setCell(23, 2); setCell(23, 6)
        setCell(25, 1); setCell(25, 2); setCell(25, 6); setCell(25, 7)
        
        // 4. Far Right Block (The catcher/stabilizer)
        setCell(35, 3); setCell(36, 3)
        setCell(35, 4); setCell(36, 4)
        
        return grid
    }
    
    static func createPulsarGrid(width: Int, height: Int) -> [UInt8] {
        var grid = [UInt8](repeating: 0, count: width * height)
        
        let cx = width / 2
        let cy = height / 2
        
        // Helper to safely set a cell
        func setCell(_ x: Int, _ y: Int) {
            let finalX = cx + x
            let finalY = cy + y
            if finalX >= 0 && finalX < width && finalY >= 0 && finalY < height {
                grid[finalY * width + finalX] = 1
            }
        }
        
        // Helper to draw a horizontal bar of length 3
        func drawHorizBar(x: Int, y: Int) {
            setCell(x, y)
            setCell(x + 1, y)
            setCell(x + 2, y)
        }
        
        // Helper to draw a vertical bar of length 3
        func drawVertBar(x: Int, y: Int) {
            setCell(x, y)
            setCell(x, y + 1)
            setCell(x, y + 2)
        }

        // --- The Pulsar Pattern ---
        // The Pulsar is formed by 8 bars on the "outside" (rows/cols ±6)
        // and 8 bars on the "inside" (rows/cols ±1)
        
        // 1. Horizontal Bars (Rows at y = -6, -1, 1, 6)
        // Left side (starts at x = -4) and Right side (starts at x = 2)
        for y in [-6, -1, 1, 6] {
            drawHorizBar(x: -4, y: y)
            drawHorizBar(x: 2, y: y)
        }
        
        // 2. Vertical Bars (Cols at x = -6, -1, 1, 6)
        // Top side (starts at y = -4) and Bottom side (starts at y = 2)
        for x in [-6, -1, 1, 6] {
            drawVertBar(x: x, y: -4)
            drawVertBar(x: x, y: 2)
        }
        
        return grid
    }
}
