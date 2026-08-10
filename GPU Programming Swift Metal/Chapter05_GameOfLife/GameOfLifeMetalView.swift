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
import MetalKit
import SwiftUI

struct GameOfLifeMetalView: NSViewRepresentable {
    let gameOfLife: GameOfLife
    
    func makeCoordinator() -> Coordinator {
        Coordinator(parent: self, gameOfLife: gameOfLife)
    }
    
    func makeNSView(context: Context) -> MTKView {
        let mtkView = MTKView()
        mtkView.device = MTLCreateSystemDefaultDevice()
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        mtkView.delegate = context.coordinator
        return mtkView
    }
    
    func updateNSView(_ nsView: MTKView, context: Context) {
        
    }
    
    class Coordinator: NSObject, MTKViewDelegate {
        var parent: GameOfLifeMetalView
        var gameOfLife: GameOfLife!
        var commandQueue: MTLCommandQueue?
        var pipelineState: MTLRenderPipelineState?
        
        init(parent: GameOfLifeMetalView, gameOfLife: GameOfLife) {
            self.parent = parent
            self.gameOfLife = gameOfLife
            super.init()
            
            let device = MTLCreateSystemDefaultDevice()!
            self.commandQueue = device.makeCommandQueue()
            
            let library = device.makeDefaultLibrary()!
            let vertexFunction = library.makeFunction(name: "vertexRenderGame")!
            let fragmentFunction = library.makeFunction(name: "fragmentRenderGame")!
            
            let pipelineDescriptor = MTLRenderPipelineDescriptor()
            pipelineDescriptor.vertexFunction = vertexFunction
            pipelineDescriptor.fragmentFunction = fragmentFunction
            pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            
            // enable point sprites
            pipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
            
            self.pipelineState = try! device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        }
        
        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            
        }
        
        struct Uniforms {
            var viewSize: SIMD2<Float>
            var zoom: Float
            // ensures the next SIMD2 aligns to 8 bytes
            var _pad: Float = 0
            var offset: SIMD2<Float>
        }
        
        func draw(in view: MTKView) {
            let commandBuffer = commandQueue!.makeCommandBuffer()!
            let renderPassDescriptor = view.currentRenderPassDescriptor!
            let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!
            
            renderEncoder.setRenderPipelineState(pipelineState!)
            
            // bind grid data
            renderEncoder.setVertexBuffer(gameOfLife.currentGridBuffer, offset: 0, index: 0)
            
            // bind grid dimensions
            var pixels = gameOfLife.gridSizePixels
            renderEncoder.setVertexBytes(&pixels, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 1)
            
            let width = Float(view.drawableSize.width)
            let height = Float(view.drawableSize.height)
            
            // zoom = 10.0 means every cell is a 10x10 block of pixels
            let desiredZoom: Float = 6.0
            
            // Center the camera
            let viewCenter = SIMD2<Float>(width / 2.0, height / 2.0)
            let gridCenter = SIMD2<Float>(Float(gameOfLife.gridSizePixels.x) / 2.0, Float(gameOfLife.gridSizePixels.y) / 2.0)
            let centeredOffset = viewCenter - (gridCenter * desiredZoom)
            
            var uniforms = Uniforms(
                viewSize: [width, height],
                zoom: desiredZoom,
                offset: centeredOffset
            )
            renderEncoder.setVertexBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 2)
            
            // draw one vertex for every cell in the grid
            let totalCells = Int(gameOfLife.gridSizePixels.x * gameOfLife.gridSizePixels.y)
            renderEncoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: totalCells)
            
            renderEncoder.endEncoding()
            
            if let drawable = view.currentDrawable {
                commandBuffer.present(drawable)
            }
            commandBuffer.commit()
        }
    }
}
