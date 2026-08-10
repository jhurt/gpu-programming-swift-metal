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

struct GrayScottMetalView: NSViewRepresentable {
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    func makeNSView(context: Context) -> MTKView {
        guard let grayScott = GrayScottController.shared.grayScott else {
            fatalError("GrayScottController.shared.grayScott is nil")
        }
        let mtkView = MTKView()
        mtkView.device = grayScott.device
        mtkView.delegate = context.coordinator
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.preferredFramesPerSecond = 60
        mtkView.isPaused = false
        mtkView.enableSetNeedsDisplay = false
        
        if let window = mtkView.window {
            NotificationCenter.default.addObserver(
                context.coordinator,
                selector: #selector(Coordinator.windowWillClose),
                name: NSWindow.willCloseNotification,
                object: window
            )
        }
        
        return mtkView
    }
    
    func updateNSView(_ nsView: MTKView, context: Context) {
        
    }
    
    class Coordinator: NSObject, MTKViewDelegate {
        var commandQueue: MTLCommandQueue?
        var pipelineState: MTLRenderPipelineState?
        
        override init() {
            guard let grayScott = GrayScottController.shared.grayScott else {
                fatalError("GrayScottController.shared.grayScott is nil")
            }
            self.commandQueue = grayScott.device.makeCommandQueue()
            
            let library = grayScott.device.makeDefaultLibrary()!
            let vertexFunction = library.makeFunction(name: "vertexRenderGrayScott")!
            let fragmentFunction = library.makeFunction(name: "fragmentRenderGrayScott")!
            
            let renderPipelineDescriptor = MTLRenderPipelineDescriptor()
            renderPipelineDescriptor.vertexFunction = vertexFunction
            renderPipelineDescriptor.fragmentFunction = fragmentFunction
            renderPipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            pipelineState = try! grayScott.device.makeRenderPipelineState(descriptor: renderPipelineDescriptor)
        }
        
        deinit {
            NotificationCenter.default.removeObserver(self)
        }
        
        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            
        }
        
        func draw(in view: MTKView) {
            guard let grayScott = GrayScottController.shared.grayScott else {
                fatalError("GrayScottController.shared.grayScott is nil")
            }
            grayScott.step()
            let commandBuffer = commandQueue!.makeCommandBuffer()!
            let renderPassDescriptor = view.currentRenderPassDescriptor!
            let renderCommandEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!
            
            renderCommandEncoder.setRenderPipelineState(pipelineState!)
            renderCommandEncoder.setVertexBuffer(grayScott.vBuffers[grayScott.currentBufferIndex], offset: 0, index: 0)
            renderCommandEncoder.setVertexBytes(&grayScott.gridSize, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 1)
            var viewSize = SIMD2<Float>(Float(view.drawableSize.width), Float(view.drawableSize.height))
            renderCommandEncoder.setVertexBytes(&viewSize, length: MemoryLayout<SIMD2<Float>>.stride, index: 2)
            renderCommandEncoder.setVertexBytes(&grayScott.environmentCount, length: MemoryLayout<UInt32>.stride, index: 3)
            renderCommandEncoder.setVertexBuffer(grayScott.cellEnvironmentsBuffer, offset: 0, index: 4)
            
            // draw one point per pixel
            renderCommandEncoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: grayScott.gridWidth * grayScott.gridHeight)
            
            renderCommandEncoder.endEncoding()
            let drawable = view.currentDrawable!
            commandBuffer.present(drawable)
            
            commandBuffer.commit()
        }
        
        @objc func windowWillClose(_ notification: Notification) {
            GrayScottController.shared.grayScott?.stop()
        }
    }
}
