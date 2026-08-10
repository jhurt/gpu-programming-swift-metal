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

import AppKit
import CoreImage
import ImageIO
import Metal
import MetalKit
import UniformTypeIdentifiers

private func saveColorTextureAsPNG(texture: MTLTexture, url: URL) {
    var image = CIImage(mtlTexture: texture, options: [.colorSpace: CGColorSpaceCreateDeviceRGB()])!
    
    // flip image vertically from Metal to standard orientation
    image = image.transformed(by: CGAffineTransform(scaleX: 1, y: -1))
        .transformed(by: CGAffineTransform(translationX: 0, y: image.extent.height))
    
    let context = CIContext()
    try! context.writePNGRepresentation(of: image, to: url, format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB())
}

private func saveGrayTextureAsPNG(texture: MTLTexture, url: URL) {
    let grayCS = CGColorSpaceCreateDeviceGray()
    var image = CIImage(
        mtlTexture: texture,
        options: [.colorSpace: grayCS]
    )!

    // flip image vertically from Metal to standard orientation
    image = image.transformed(by: CGAffineTransform(scaleX: 1, y: -1))
        .transformed(by: CGAffineTransform(translationX: 0, y: image.extent.height))
    
    let context = CIContext()
    try! context.writePNGRepresentation(of: image, to: url, format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB())
}

private func fftCooleyTukey(imageIn: String, imageOut: String, useSIMD: Bool = false) async {
    let processor = ParallelProcessor(log: true)
    
    // (1) load an RGB image from disk and create a `MTLTexture` instance from it
    let textureLoader = MTKTextureLoader(device: processor.device)
    let imageURL = URL(fileURLWithPath: imageIn)
    let inputTexture = try! await textureLoader.newTexture(URL: imageURL, options: [.textureUsage: MTLTextureUsage.shaderRead.rawValue, .SRGB: false])
    
    // (2) create an output `MTLTexture` to visualize the filtered result
    let outputTextureDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm,
                                                                           width: inputTexture.width,
                                                                           height: inputTexture.height,
                                                                           mipmapped: false)
    outputTextureDescriptor.usage = [.shaderRead, .shaderWrite]
    let outputTexture = processor.device.makeTexture(descriptor: outputTextureDescriptor)!
    
    // (3) setup the buffers for the kernels
    let pixelCount = inputTexture.width * inputTexture.height
    let outputBuffer1 = processor.device.makeBuffer(length: MemoryLayout<SIMD2<Float>>.stride * pixelCount, options: .storageModeShared)!
    
    var width = UInt32(inputTexture.width)
    let widthBuffer = processor.device.makeBuffer(bytes: &width, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    
    var height = UInt32(inputTexture.height)
    let heightBuffer = processor.device.makeBuffer(bytes: &height, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!

    var fftAngleMultiplier: Float = -2.0
    let fftAngleMultiplierBuffer = processor.device.makeBuffer(bytes: &fftAngleMultiplier, length: MemoryLayout<Float>.stride, options: .storageModeShared)!

    var ifftAngleMultiplier: Float = 2.0
    let ifftAngleMultiplierBuffer = processor.device.makeBuffer(bytes: &ifftAngleMultiplier, length: MemoryLayout<Float>.stride, options: .storageModeShared)!

    var normalizationFactor: Float = 1.0
    let normalizationFactorBuffer = processor.device.makeBuffer(bytesNoCopy: &normalizationFactor, length: MemoryLayout<Float>.stride, options: .storageModeShared)!

    let bufferY = processor.device.makeBuffer(length: MemoryLayout<SIMD2<Float>>.stride * pixelCount, options: .storageModeShared)!
    let bufferCr = processor.device.makeBuffer(length: MemoryLayout<Float>.stride * pixelCount, options: .storageModeShared)!
    let bufferCb = processor.device.makeBuffer(length: MemoryLayout<Float>.stride * pixelCount, options: .storageModeShared)!

    // (1) RGB to YCrCb
    let optionsRgbToYCrCb = ProcessOptions(threadsPerGridMTLSize: MTLSizeMake(inputTexture.width, inputTexture.height, 1),
                                           threadsPerThreadgroupMTLSize: MTLSizeMake(16, 16, 1),
                                           removeResidencySet: false, resetCommandAllocator: false)
    await processor.process(kernelName: "rgbToYCrCb", buffers: bufferY, bufferCr, bufferCb, textures: inputTexture, options: optionsRgbToYCrCb)
    let outputBuffer2 = bufferY
    
    // (2) fft 1
    let fftKernel = useSIMD ? "fftCooleyTukeySIMD" : "fftCooleyTukey"
    await processor.process(kernelName: fftKernel,
                            buffers: outputBuffer2, widthBuffer, fftAngleMultiplierBuffer, normalizationFactorBuffer, outputBuffer1,
                            threadgroupMemory: MemoryLayout<SIMD2<Float>>.stride * inputTexture.width,
                            options: ProcessOptions(
                                threadgroupsPerGridMTLSize: MTLSizeMake(1, inputTexture.height, 1),
                                threadsPerThreadgroupMTLSize: MTLSizeMake(inputTexture.width / 2, 1, 1),
                                removeResidencySet: false, resetCommandAllocator: false))
    
    // (3) transpose 1
    await processor.process(kernelName: "fftTranspose",
                            buffers: outputBuffer1, widthBuffer, heightBuffer, outputBuffer2,
                            options: ProcessOptions(
                                threadsPerGridMTLSize: MTLSizeMake(inputTexture.width, inputTexture.height, 1),
                                threadsPerThreadgroupMTLSize: MTLSizeMake(32, 32, 1),
                                removeResidencySet: false, resetCommandAllocator: false))
    
    // (4) fft 2
    await processor.process(kernelName: fftKernel,
                            buffers: outputBuffer2, heightBuffer, fftAngleMultiplierBuffer, normalizationFactorBuffer, outputBuffer1,
                            threadgroupMemory: MemoryLayout<SIMD2<Float>>.stride * inputTexture.height,
                            options: ProcessOptions(
                                threadgroupsPerGridMTLSize: MTLSizeMake(1, inputTexture.width, 1),
                                threadsPerThreadgroupMTLSize: MTLSizeMake(inputTexture.height / 2, 1, 1),
                                removeResidencySet: false, resetCommandAllocator: false))
    
    // (5) transpose 2
    await processor.process(kernelName: "fftTranspose",
                            buffers: outputBuffer1, heightBuffer, widthBuffer, outputBuffer2,
                            options: ProcessOptions(
                                threadsPerGridMTLSize: MTLSizeMake(inputTexture.height, inputTexture.width, 1),
                                threadsPerThreadgroupMTLSize: MTLSizeMake(32, 32, 1),
                                removeResidencySet: false, resetCommandAllocator: false))

    // (1) filter
    let optionsFilter = ProcessOptions(threadsPerGridMTLSize: MTLSizeMake(inputTexture.height, inputTexture.width, 1),
                                       threadsPerThreadgroupMTLSize: MTLSizeMake(16, 16,  1),
                                       removeResidencySet: false, resetCommandAllocator: false)
    await processor.process(kernelName: "fftGaussianBlur", buffers: outputBuffer2, heightBuffer, widthBuffer, options: optionsFilter)
    
    // (2) ifft 1
    await processor.process(kernelName: fftKernel,
                            buffers: outputBuffer2, widthBuffer, ifftAngleMultiplierBuffer, normalizationFactorBuffer, outputBuffer1,
                            threadgroupMemory: MemoryLayout<SIMD2<Float>>.stride * inputTexture.width,
                            options: ProcessOptions(
                                threadgroupsPerGridMTLSize: MTLSizeMake(1, inputTexture.height, 1),
                                threadsPerThreadgroupMTLSize: MTLSizeMake(inputTexture.width / 2, 1, 1),
                                removeResidencySet: false, resetCommandAllocator: false))
    
    // (3) transpose 3
    await processor.process(kernelName: "fftTranspose",
                            buffers: outputBuffer1, heightBuffer, widthBuffer, outputBuffer2,
                            options: ProcessOptions(
                                threadsPerGridMTLSize: MTLSizeMake(inputTexture.height, inputTexture.width, 1),
                                threadsPerThreadgroupMTLSize: MTLSizeMake(32, 32, 1),
                                removeResidencySet: false, resetCommandAllocator: false))
    
    // (4) ifft 2
    normalizationFactor = 1.0 / (Float(inputTexture.width * inputTexture.height))
    await processor.process(kernelName: fftKernel,
                            buffers: outputBuffer2, heightBuffer, ifftAngleMultiplierBuffer, normalizationFactorBuffer, outputBuffer1,
                            threadgroupMemory: MemoryLayout<SIMD2<Float>>.stride * inputTexture.height,
                            options: ProcessOptions(
                                threadgroupsPerGridMTLSize: MTLSizeMake(1, inputTexture.width, 1),
                                threadsPerThreadgroupMTLSize: MTLSizeMake(inputTexture.height / 2, 1, 1),
                                removeResidencySet: false, resetCommandAllocator: false))
    
    // (5) transpose 4
    await processor.process(kernelName: "fftTranspose",
                            buffers: outputBuffer1, heightBuffer, widthBuffer, outputBuffer2,
                            options: ProcessOptions(
                                threadsPerGridMTLSize: MTLSizeMake(inputTexture.height, inputTexture.width, 1),
                                threadsPerThreadgroupMTLSize: MTLSizeMake(32, 32, 1),
                                removeResidencySet: false, resetCommandAllocator: false))
    
    // (6) YCrCb to RGB
    let optionsYCrCbToRgb =  ProcessOptions(threadsPerGridMTLSize: MTLSizeMake(inputTexture.width, inputTexture.height, 1),
                                            threadsPerThreadgroupMTLSize: MTLSizeMake(16, 16,  1))
    await processor.process(kernelName: "yCrCbToRgb", buffers: outputBuffer2, bufferCr, bufferCb, textures: outputTexture, options: optionsYCrCbToRgb)
    
    // (7) save the image to disk
    let outputURL = URL(fileURLWithPath: imageOut)
    saveColorTextureAsPNG(texture: outputTexture, url: outputURL)
}

private func fftCooleyTukeyBarrier(imageIn: String, imageOut: String, useSIMD: Bool = false) async {
    let device = MTLCreateSystemDefaultDevice()!
    let library = try! device.makeDefaultLibrary(bundle: .main)

    let textureLoader = MTKTextureLoader(device: device)
    let imageURL = URL(fileURLWithPath: imageIn)
    let inputTexture = try! await textureLoader.newTexture(URL: imageURL, options: [.textureUsage: MTLTextureUsage.shaderRead.rawValue, .SRGB: false])
    
    let outputTextureDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm,
                                                                           width: inputTexture.width,
                                                                           height: inputTexture.height,
                                                                           mipmapped: false)
    outputTextureDescriptor.usage = [.shaderRead, .shaderWrite]
    let outputTexture = device.makeTexture(descriptor: outputTextureDescriptor)!
    
    let pixelCount = inputTexture.width * inputTexture.height
    
    let outputBuffer1 = device.makeBuffer(length: MemoryLayout<SIMD2<Float>>.stride * pixelCount, options: .storageModeShared)!
    let outputBuffer2 = device.makeBuffer(length: MemoryLayout<SIMD2<Float>>.stride * pixelCount, options: .storageModeShared)!
    
    var width = UInt32(inputTexture.width)
    let widthBuffer = device.makeBuffer(bytes: &width, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    
    var height = UInt32(inputTexture.height)
    let heightBuffer = device.makeBuffer(bytes: &height, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    
    var fftAngleMultiplier: Float = -2.0
    let fftAngleMultiplierBuffer = device.makeBuffer(bytes: &fftAngleMultiplier, length: MemoryLayout<Float>.stride, options: .storageModeShared)!

    var ifftAngleMultiplier: Float = 2.0
    let ifftAngleMultiplierBuffer = device.makeBuffer(bytes: &ifftAngleMultiplier, length: MemoryLayout<Float>.stride, options: .storageModeShared)!

    var normalizationFactor1: Float = 1.0
    let normalizationFactorBuffer1 = device.makeBuffer(bytes: &normalizationFactor1, length: MemoryLayout<Float>.stride, options: .storageModeShared)!

    var normalizationFactor2: Float = 1.0 / (Float(inputTexture.width * inputTexture.height))
    let normalizationFactorBuffer2 = device.makeBuffer(bytes: &normalizationFactor2, length: MemoryLayout<Float>.stride, options: .storageModeShared)!

    let bufferY = device.makeBuffer(length: MemoryLayout<SIMD2<Float>>.stride * pixelCount, options: .storageModeShared)!
    let bufferCr = device.makeBuffer(length: MemoryLayout<Float>.stride * pixelCount, options: .storageModeShared)!
    let bufferCb = device.makeBuffer(length: MemoryLayout<Float>.stride * pixelCount, options: .storageModeShared)!
    
    let residencySet = try! device.makeResidencySet(descriptor: .init())
    residencySet.addAllocation(outputBuffer1)
    residencySet.addAllocation(outputBuffer2)
    residencySet.addAllocation(widthBuffer)
    residencySet.addAllocation(heightBuffer)
    residencySet.addAllocation(fftAngleMultiplierBuffer)
    residencySet.addAllocation(ifftAngleMultiplierBuffer)
    residencySet.addAllocation(normalizationFactorBuffer1)
    residencySet.addAllocation(normalizationFactorBuffer2)
    residencySet.addAllocation(bufferY)
    residencySet.addAllocation(bufferCr)
    residencySet.addAllocation(bufferCb)
    residencySet.addAllocation(inputTexture)
    residencySet.addAllocation(outputTexture)
    residencySet.commit()
    
    let commandAllocator = device.makeCommandAllocator()!
    let commandBuffer = device.makeCommandBuffer()!
    commandBuffer.beginCommandBuffer(allocator: commandAllocator, options: .init())
    
    // (1) RGB to YCrCb
    let argumentTableDescriptorRgbToYCrCb = MTL4ArgumentTableDescriptor()
    argumentTableDescriptorRgbToYCrCb.maxBufferBindCount = 3
    argumentTableDescriptorRgbToYCrCb.maxTextureBindCount = 1
    
    let argumentTableRgbToYCrCb = try! device.makeArgumentTable(descriptor: argumentTableDescriptorRgbToYCrCb)
    argumentTableRgbToYCrCb.setAddress(bufferY.gpuAddress, index: 0)
    argumentTableRgbToYCrCb.setAddress(bufferCr.gpuAddress, index: 1)
    argumentTableRgbToYCrCb.setAddress(bufferCb.gpuAddress, index: 2)
    argumentTableRgbToYCrCb.setTexture(inputTexture.gpuResourceID, index: 0)
    
    let computeCommandEncoderRgbToYCrCb = commandBuffer.makeComputeCommandEncoder()!
    let kernelFunctionRgbToYCrCb = library.makeFunction(name: "rgbToYCrCb")!
    let computePipelineStateRgbToYCrCb = try! await device.makeComputePipelineState(function: kernelFunctionRgbToYCrCb)
    computeCommandEncoderRgbToYCrCb.setComputePipelineState(computePipelineStateRgbToYCrCb)
    computeCommandEncoderRgbToYCrCb.setArgumentTable(argumentTableRgbToYCrCb)
    computeCommandEncoderRgbToYCrCb.barrier(afterStages: .dispatch, beforeQueueStages: .dispatch)
    computeCommandEncoderRgbToYCrCb.dispatchThreads(threadsPerGrid: MTLSizeMake(inputTexture.width, inputTexture.height, 1),
                                                    threadsPerThreadgroup: MTLSizeMake(16, 16,  1))
    computeCommandEncoderRgbToYCrCb.endEncoding()
    
    // (2) fft 1
    let argumentTableDescriptorFFT = MTL4ArgumentTableDescriptor()
    argumentTableDescriptorFFT.maxBufferBindCount = 5
    
    let argumentTableFFT1 = try! device.makeArgumentTable(descriptor: argumentTableDescriptorFFT)
    argumentTableFFT1.setAddress(bufferY.gpuAddress, index: 0)
    argumentTableFFT1.setAddress(widthBuffer.gpuAddress, index: 1)
    argumentTableFFT1.setAddress(fftAngleMultiplierBuffer.gpuAddress, index: 2)
    argumentTableFFT1.setAddress(normalizationFactorBuffer1.gpuAddress, index: 3)
    argumentTableFFT1.setAddress(outputBuffer1.gpuAddress, index: 4)
    
    let computeCommandEncoderFFT1 = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoderFFT1.setThreadgroupMemoryLength(MemoryLayout<SIMD2<Float>>.stride * inputTexture.width, index: 0)
    let kernelFunctionFFT = library.makeFunction(name: useSIMD ? "fftCooleyTukeySIMD" : "fftCooleyTukey")!
    let computePipelineStateFFT1 = try! await device.makeComputePipelineState(function: kernelFunctionFFT)
    computeCommandEncoderFFT1.setComputePipelineState(computePipelineStateFFT1)
    computeCommandEncoderFFT1.setArgumentTable(argumentTableFFT1)
    computeCommandEncoderFFT1.barrier(afterStages: .dispatch, beforeQueueStages: .dispatch)
    computeCommandEncoderFFT1.dispatchThreadgroups(threadgroupsPerGrid: MTLSizeMake(1, inputTexture.height, 1),
                                                   threadsPerThreadgroup: MTLSizeMake(inputTexture.width / 2, 1, 1))
    computeCommandEncoderFFT1.endEncoding()
    
    // (3) transpose 1
    let argumentTableDescriptorTranspose = MTL4ArgumentTableDescriptor()
    argumentTableDescriptorTranspose.maxBufferBindCount = 4
    
    let argumentTableTranspose1 = try! device.makeArgumentTable(descriptor: argumentTableDescriptorTranspose)
    argumentTableTranspose1.setAddress(outputBuffer1.gpuAddress, index: 0)
    argumentTableTranspose1.setAddress(widthBuffer.gpuAddress, index: 1)
    argumentTableTranspose1.setAddress(heightBuffer.gpuAddress, index: 2)
    argumentTableTranspose1.setAddress(outputBuffer2.gpuAddress, index: 3)
    
    let computeCommandEncoderTranspose1 = commandBuffer.makeComputeCommandEncoder()!
    let kernelFunctionTranspose = library.makeFunction(name: "fftTranspose")!
    let computePipelineStateTranspose1 = try! await device.makeComputePipelineState(function: kernelFunctionTranspose)
    computeCommandEncoderTranspose1.setComputePipelineState(computePipelineStateTranspose1)
    computeCommandEncoderTranspose1.setArgumentTable(argumentTableTranspose1)
    computeCommandEncoderTranspose1.barrier(afterStages: .dispatch, beforeQueueStages: .dispatch)
    computeCommandEncoderTranspose1.dispatchThreads(threadsPerGrid: MTLSizeMake(inputTexture.width, inputTexture.height, 1),
                                                    threadsPerThreadgroup: MTLSizeMake(32, 32, 1))
    computeCommandEncoderTranspose1.endEncoding()
    
    // (4) fft 2
    let argumentTableFFT2 = try! device.makeArgumentTable(descriptor: argumentTableDescriptorFFT)
    argumentTableFFT2.setAddress(outputBuffer2.gpuAddress, index: 0)
    argumentTableFFT2.setAddress(heightBuffer.gpuAddress, index: 1)
    argumentTableFFT2.setAddress(fftAngleMultiplierBuffer.gpuAddress, index: 2)
    argumentTableFFT2.setAddress(normalizationFactorBuffer1.gpuAddress, index: 3)
    argumentTableFFT2.setAddress(outputBuffer1.gpuAddress, index: 4)
    
    let computeCommandEncoderFFT2 = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoderFFT2.setThreadgroupMemoryLength(MemoryLayout<SIMD2<Float>>.stride * inputTexture.height, index: 0)
    let computePipelineStateFFT2 = try! await device.makeComputePipelineState(function: kernelFunctionFFT)
    computeCommandEncoderFFT2.setComputePipelineState(computePipelineStateFFT2)
    computeCommandEncoderFFT2.setArgumentTable(argumentTableFFT2)
    computeCommandEncoderFFT2.barrier(afterStages: .dispatch, beforeQueueStages: .dispatch)
    computeCommandEncoderFFT2.dispatchThreadgroups(threadgroupsPerGrid: MTLSizeMake(1, inputTexture.width, 1),
                                                   threadsPerThreadgroup:  MTLSizeMake(inputTexture.height / 2, 1, 1))
    computeCommandEncoderFFT2.endEncoding()
    
    // (1) filter
    let argumentTableDescriptorFilter = MTL4ArgumentTableDescriptor()
    argumentTableDescriptorFilter.maxBufferBindCount = 3
    
    let argumentTableFilter = try! device.makeArgumentTable(descriptor: argumentTableDescriptorFFT)
    argumentTableFilter.setAddress(outputBuffer1.gpuAddress, index: 0)
    argumentTableFilter.setAddress(heightBuffer.gpuAddress, index: 1)
    argumentTableFilter.setAddress(widthBuffer.gpuAddress, index: 2)
    
    let computeCommandEncoderFilter = commandBuffer.makeComputeCommandEncoder()!
    let kernelFunctionFilter = library.makeFunction(name: "fftGaussianBlur")!
    let computePipelineStateFilter = try! await device.makeComputePipelineState(function: kernelFunctionFilter)
    computeCommandEncoderFilter.setComputePipelineState(computePipelineStateFilter)
    computeCommandEncoderFilter.setArgumentTable(argumentTableFilter)
    computeCommandEncoderFilter.barrier(afterStages: .dispatch, beforeQueueStages: .dispatch)
    computeCommandEncoderFilter.dispatchThreads(threadsPerGrid: MTLSizeMake(inputTexture.height, inputTexture.width, 1),
                                                threadsPerThreadgroup: MTLSizeMake(16, 16,  1))
    computeCommandEncoderFilter.endEncoding()
    
    // (2) ifft 1
    let argumentTableIFFT1 = try! device.makeArgumentTable(descriptor: argumentTableDescriptorFFT)
    argumentTableIFFT1.setAddress(outputBuffer1.gpuAddress, index: 0)
    argumentTableIFFT1.setAddress(heightBuffer.gpuAddress, index: 1)
    argumentTableIFFT1.setAddress(ifftAngleMultiplierBuffer.gpuAddress, index: 2)
    argumentTableIFFT1.setAddress(normalizationFactorBuffer1.gpuAddress, index: 3)
    argumentTableIFFT1.setAddress(outputBuffer2.gpuAddress, index: 4)
    
    let computeCommandEncoderIFFT1 = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoderIFFT1.setThreadgroupMemoryLength(MemoryLayout<SIMD2<Float>>.stride * inputTexture.height, index: 0)
    let computePipelineStateIFFT1 = try! await device.makeComputePipelineState(function: kernelFunctionFFT)
    computeCommandEncoderIFFT1.setComputePipelineState(computePipelineStateIFFT1)
    computeCommandEncoderIFFT1.setArgumentTable(argumentTableIFFT1)
    computeCommandEncoderIFFT1.barrier(afterStages: .dispatch, beforeQueueStages: .dispatch)
    computeCommandEncoderIFFT1.dispatchThreadgroups(threadgroupsPerGrid: MTLSizeMake(1, inputTexture.width, 1),
                                                    threadsPerThreadgroup: MTLSizeMake(inputTexture.height / 2, 1, 1))
    computeCommandEncoderIFFT1.endEncoding()
    
    // (3) transpose 2
    let argumentTableTranspose2 = try! device.makeArgumentTable(descriptor: argumentTableDescriptorTranspose)
    argumentTableTranspose2.setAddress(outputBuffer2.gpuAddress, index: 0)
    argumentTableTranspose2.setAddress(heightBuffer.gpuAddress, index: 1)
    argumentTableTranspose2.setAddress(widthBuffer.gpuAddress, index: 2)
    argumentTableTranspose2.setAddress(outputBuffer1.gpuAddress, index: 3)
    
    let computeCommandEncoderTranspose2 = commandBuffer.makeComputeCommandEncoder()!
    let computePipelineStateTranspose2 = try! await device.makeComputePipelineState(function: kernelFunctionTranspose)
    computeCommandEncoderTranspose2.setComputePipelineState(computePipelineStateTranspose2)
    computeCommandEncoderTranspose2.setArgumentTable(argumentTableTranspose2)
    computeCommandEncoderTranspose2.barrier(afterStages: .dispatch, beforeQueueStages: .dispatch)
    computeCommandEncoderTranspose2.dispatchThreads(threadsPerGrid: MTLSizeMake(inputTexture.height, inputTexture.width, 1),
                                                    threadsPerThreadgroup: MTLSizeMake(32, 32, 1))
    computeCommandEncoderTranspose2.endEncoding()
    
    // (4) ifft 2
    let argumentTableIFFT2 = try! device.makeArgumentTable(descriptor: argumentTableDescriptorFFT)
    argumentTableIFFT2.setAddress(outputBuffer1.gpuAddress, index: 0)
    argumentTableIFFT2.setAddress(widthBuffer.gpuAddress, index: 1)
    argumentTableIFFT2.setAddress(ifftAngleMultiplierBuffer.gpuAddress, index: 2)
    argumentTableIFFT2.setAddress(normalizationFactorBuffer2.gpuAddress, index: 3)
    argumentTableIFFT2.setAddress(outputBuffer2.gpuAddress, index: 4)
    
    let computeCommandEncoderIFFT2 = commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoderIFFT2.setThreadgroupMemoryLength(MemoryLayout<SIMD2<Float>>.stride * inputTexture.width, index: 0)
    let computePipelineStateIFFT2 = try! await device.makeComputePipelineState(function: kernelFunctionFFT)
    computeCommandEncoderIFFT2.setComputePipelineState(computePipelineStateIFFT2)
    computeCommandEncoderIFFT2.setArgumentTable(argumentTableIFFT2)
    computeCommandEncoderIFFT2.barrier(afterStages: .dispatch, beforeQueueStages: .dispatch)
    computeCommandEncoderIFFT2.dispatchThreadgroups(threadgroupsPerGrid: MTLSizeMake(1, inputTexture.height, 1),
                                                    threadsPerThreadgroup: MTLSizeMake(inputTexture.width / 2, 1, 1))
    computeCommandEncoderIFFT2.endEncoding()
    
    // (5) YCrCb to RGB
    let argumentTableDescriptorYCrCbToRgb = MTL4ArgumentTableDescriptor()
    argumentTableDescriptorYCrCbToRgb.maxBufferBindCount = 3
    argumentTableDescriptorYCrCbToRgb.maxTextureBindCount = 1
    
    let argumentTableYCrCbToRgb = try! device.makeArgumentTable(descriptor: argumentTableDescriptorYCrCbToRgb)
    argumentTableYCrCbToRgb.setAddress(outputBuffer2.gpuAddress, index: 0)
    argumentTableYCrCbToRgb.setAddress(bufferCr.gpuAddress, index: 1)
    argumentTableYCrCbToRgb.setAddress(bufferCb.gpuAddress, index: 2)
    argumentTableYCrCbToRgb.setTexture(outputTexture.gpuResourceID, index: 0)
    
    let computeCommandEncoderYCrCbToRgb = commandBuffer.makeComputeCommandEncoder()!
    let kernelFunctionYCrCbToRgb = library.makeFunction(name: "yCrCbToRgb")!
    let computePipelineStateYCrCbToRgb = try! await device.makeComputePipelineState(function: kernelFunctionYCrCbToRgb)
    computeCommandEncoderYCrCbToRgb.setComputePipelineState(computePipelineStateYCrCbToRgb)
    computeCommandEncoderYCrCbToRgb.setArgumentTable(argumentTableYCrCbToRgb)
    computeCommandEncoderYCrCbToRgb.dispatchThreads(threadsPerGrid: MTLSizeMake(inputTexture.width, inputTexture.height, 1),
                                                    threadsPerThreadgroup: MTLSizeMake(16, 16,  1))
    computeCommandEncoderYCrCbToRgb.endEncoding()
    
    commandBuffer.endCommandBuffer()
    
    let commandQueue = device.makeMTL4CommandQueue()!
    commandQueue.addResidencySet(residencySet)
    let commitOptions = MTL4CommitOptions()
    
    try! await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
        commitOptions.addFeedbackHandler { feedback in
            if let error = feedback.error {
                continuation.resume(throwing: error)
            } else {
                continuation.resume()
            }
        }
        commandQueue.commit([commandBuffer], options: commitOptions)
    }
    
    let outputURL = URL(fileURLWithPath: imageOut)
    saveColorTextureAsPNG(texture: outputTexture, url: outputURL)
}

private func fftStockham(imageIn: String, imageOut: String, isRadix4: Bool = false) async {
    let processor = ParallelProcessor(log: true)
    
    let textureLoader = MTKTextureLoader(device: processor.device)
    let imageURL = URL(fileURLWithPath: imageIn)
    let inputTexture = try! await textureLoader.newTexture(URL: imageURL, options: [.textureUsage: MTLTextureUsage.shaderRead.rawValue, .SRGB: false])
    
    let outputTextureDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm,
                                                                           width: inputTexture.width,
                                                                           height: inputTexture.height,
                                                                           mipmapped: false)
    outputTextureDescriptor.usage = [.shaderRead, .shaderWrite]
    let outputTexture = processor.device.makeTexture(descriptor: outputTextureDescriptor)!

    let pixelCount = inputTexture.width * inputTexture.height
    
    let bufferY = processor.device.makeBuffer(length: MemoryLayout<SIMD2<Float>>.stride * pixelCount, options: .storageModeShared)!
    let bufferCr = processor.device.makeBuffer(length: MemoryLayout<Float>.stride * pixelCount, options: .storageModeShared)!
    let bufferCb = processor.device.makeBuffer(length: MemoryLayout<Float>.stride * pixelCount, options: .storageModeShared)!

    var width = UInt32(inputTexture.width)
    let widthBuffer = processor.device.makeBuffer(bytes: &width, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    
    var height = UInt32(inputTexture.height)
    let heightBuffer = processor.device.makeBuffer(bytes: &height, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    
    var fftAngleMultiplier: Float = -2.0
    let fftAngleMultiplierBuffer = processor.device.makeBuffer(bytes: &fftAngleMultiplier, length: MemoryLayout<Float>.stride, options: .storageModeShared)!
    
    var ifftAngleMultiplier: Float = 2.0
    let ifftAngleMultiplierBuffer = processor.device.makeBuffer(bytes: &ifftAngleMultiplier, length: MemoryLayout<Float>.stride, options: .storageModeShared)!
    
    var normalizationFactor: Float = 1.0
    let normalizationFactorBuffer = processor.device.makeBuffer(bytesNoCopy: &normalizationFactor, length: MemoryLayout<Float>.stride, options: .storageModeShared)!
    
    var inputBuffer = bufferY
    var outputBuffer = processor.device.makeBuffer(length: MemoryLayout<SIMD2<Float>>.stride * pixelCount, options: .storageModePrivate)!
    
    let threadsPerThreadgroup = MTLSizeMake(16, 16,  1)
    let fftKernel = isRadix4 ? "fftStockhamRadix4" : "fftStockham"
    let radix = isRadix4 ? 4 : 2
    
    // (1) RGB to YCrCb
    let optionsRgbToYCrCb = ProcessOptions(threadsPerGridMTLSize: MTLSizeMake(inputTexture.width, inputTexture.height, 1),
                                           threadsPerThreadgroupMTLSize: MTLSizeMake(16, 16,  1),
                                           removeResidencySet: false, resetCommandAllocator: false)
    await processor.process(kernelName: "rgbToYCrCb", buffers: bufferY, bufferCr, bufferCb, textures: inputTexture, options: optionsRgbToYCrCb)

    // (2) forward horizontal passes
    var logWidth = Int(log2(Double(inputTexture.width)))
    if isRadix4 {
        logWidth /= 2
    }
    for i in 0..<logWidth {
        var iteration = UInt32(i)
        let iterationBuffer = processor.device.makeBuffer(bytes: &iteration, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        
        // width/radix butterflies per row
        await processor.process(kernelName: fftKernel,
                                buffers: inputBuffer, outputBuffer, iterationBuffer, widthBuffer, fftAngleMultiplierBuffer, normalizationFactorBuffer,
                                options: ProcessOptions(threadsPerGridMTLSize: MTLSizeMake(inputTexture.width / radix, inputTexture.height, 1),
                                                        threadsPerThreadgroupMTLSize: threadsPerThreadgroup,
                                                        removeResidencySet: false, resetCommandAllocator: false))
        
        swap(&inputBuffer, &outputBuffer)
    }
    
    // (3) transpose 1
    await processor.process(kernelName: "fftTranspose",
                            buffers: inputBuffer, widthBuffer, heightBuffer, outputBuffer,
                            options: ProcessOptions(
                                threadsPerGridMTLSize: MTLSizeMake(inputTexture.width, inputTexture.height, 1),
                                threadsPerThreadgroupMTLSize: MTLSizeMake(32, 32, 1),
                                removeResidencySet: false, resetCommandAllocator: false))
    swap(&inputBuffer, &outputBuffer)
    
    // (4) forward vertical passes
    var logHeight = Int(log2(Double(inputTexture.height)))
    if isRadix4 {
        logHeight /= 2
    }
    for i in 0..<logHeight {
        var iteration = UInt32(i)
        let iterationBuffer = processor.device.makeBuffer(bytes: &iteration, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        
        // height/radix butterflies per column
        await processor.process(kernelName: fftKernel,
                                buffers: inputBuffer, outputBuffer, iterationBuffer, heightBuffer, fftAngleMultiplierBuffer, normalizationFactorBuffer,
                                options: ProcessOptions(threadsPerGridMTLSize: MTLSizeMake(inputTexture.height / radix, inputTexture.width, 1),
                                                        threadsPerThreadgroupMTLSize: threadsPerThreadgroup,
                                                        removeResidencySet: false, resetCommandAllocator: false))
        
        swap(&inputBuffer, &outputBuffer)
    }
    
    // (1) filter
    let optionsFilter = ProcessOptions(threadsPerGridMTLSize: MTLSizeMake(inputTexture.height, inputTexture.width, 1),
                                       threadsPerThreadgroupMTLSize: MTLSizeMake(16, 16,  1),
                                       removeResidencySet: false, resetCommandAllocator: false)
    await processor.process(kernelName: "fftGaussianBlur", buffers: inputBuffer, heightBuffer, widthBuffer, options: optionsFilter)
    
    // (2) inverse vertical passes
    for i in 0..<logHeight {
        var iteration = UInt32(i)
        let iterationBuffer = processor.device.makeBuffer(bytes: &iteration, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        
        // height/radix butterflies per column
        await processor.process(kernelName: fftKernel,
                                buffers: inputBuffer, outputBuffer, iterationBuffer, heightBuffer, ifftAngleMultiplierBuffer, normalizationFactorBuffer,
                                options: ProcessOptions(threadsPerGridMTLSize: MTLSizeMake(inputTexture.height / radix, inputTexture.width, 1),
                                                        threadsPerThreadgroupMTLSize: threadsPerThreadgroup,
                                                        removeResidencySet: false, resetCommandAllocator: false))
        
        swap(&inputBuffer, &outputBuffer)
    }
    
    // (3) transpose 2
    await processor.process(kernelName: "fftTranspose",
                            buffers: inputBuffer, heightBuffer, widthBuffer, outputBuffer,
                            options: ProcessOptions(
                                threadsPerGridMTLSize: MTLSizeMake(inputTexture.height, inputTexture.width, 1),
                                threadsPerThreadgroupMTLSize: MTLSizeMake(32, 32, 1),
                                removeResidencySet: false, resetCommandAllocator: false))
    swap(&inputBuffer, &outputBuffer)
    
    // (4) inverse horizontal passes
    for i in 0..<logWidth {
        var iteration = UInt32(i)
        let iterationBuffer = processor.device.makeBuffer(bytes: &iteration, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        
        if i == logWidth - 1 {
            normalizationFactor = 1.0 / (Float(inputTexture.width * inputTexture.height))
        }
        
        // width/radix butterflies per row
        await processor.process(kernelName: fftKernel,
                                buffers: inputBuffer, outputBuffer, iterationBuffer, widthBuffer, ifftAngleMultiplierBuffer, normalizationFactorBuffer,
                                options: ProcessOptions(threadsPerGridMTLSize: MTLSizeMake(inputTexture.width / radix, inputTexture.height, 1),
                                                        threadsPerThreadgroupMTLSize: threadsPerThreadgroup,
                                                        removeResidencySet: false, resetCommandAllocator: false))
        
        swap(&inputBuffer, &outputBuffer)
    }
    
    // (5) YCrCb to RGB
    let optionsYCrCbToRgb =  ProcessOptions(threadsPerGridMTLSize: MTLSizeMake(inputTexture.width, inputTexture.height, 1),
                                            threadsPerThreadgroupMTLSize: MTLSizeMake(16, 16,  1))
    await processor.process(kernelName: "yCrCbToRgb", buffers: inputBuffer, bufferCr, bufferCb, textures: outputTexture, options: optionsYCrCbToRgb)
    
    let outputURL = URL(fileURLWithPath: imageOut)
    saveColorTextureAsPNG(texture: outputTexture, url: outputURL)
}

func fastFourierTransform() async {
    let openPanel = NSOpenPanel()
    
    openPanel.title = "Select an Image"
    openPanel.showsHiddenFiles = false
    openPanel.allowsMultipleSelection = false
    openPanel.canChooseDirectories = false
    openPanel.allowedContentTypes = [.image]
    openPanel.begin { response in
        if response == .OK {
            guard let selectedUrl = openPanel.url else { return }
            let path = selectedUrl.path
            Task {
                await fftCooleyTukey(imageIn: path, imageOut: "\(path.dropLast(4))_cooley_tukey_blurred.png", useSIMD: true)
                await fftCooleyTukey(imageIn: path, imageOut: "\(path.dropLast(4))_cooley_tukey_simd_blurred.png")
                await fftStockham(imageIn: path, imageOut: "\(path.dropLast(4))_stockham_blurred.png")
                await fftStockham(imageIn: path, imageOut: "\(path.dropLast(4))_stockham_radix4_blurred.png", isRadix4: true)
            }
        }
    }
}
