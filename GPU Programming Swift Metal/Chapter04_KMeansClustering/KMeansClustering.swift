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
import MetalKit
import simd

typealias Point2D = SIMD2<Float>

nonisolated private func sequential(points: [Point2D], k: Int, maxIterations: Int) -> [UInt32] {
    // (1) initially assign the centroids to the first k points
    var centroids:[Point2D] = []
    for i in 0..<k {
        centroids.append(points[i])
    }

    // (2) track the cluster index for each point
    var clusterIndices: [UInt32] = Array(repeating: 0, count: points.count)

    for iteration in 0..<maxIterations {
        var clusterPointSums = [Point2D](repeating: Point2D(0, 0), count: k)
        var clusterSizes = [Int](repeating: 0, count: k)
        
        for i in 0..<points.count {
            // (3) determine the closest cluster for this point
            var minimumDistance = Float.infinity
            var closestCentroidIndex = -1
            for j in 0..<k {
                let distance = simd_fast_distance(points[i], centroids[j])
                if distance < minimumDistance {
                    minimumDistance = distance
                    closestCentroidIndex = j
                }
            }
            
            // (4) update clusters sizes and cluster point sums
            if closestCentroidIndex != -1 {
                clusterPointSums[closestCentroidIndex] += points[i]
                clusterSizes[closestCentroidIndex] += 1
                clusterIndices[i] = UInt32(closestCentroidIndex)
            }
        }
        
        // (5) calculate centroids for each cluster
        var converged = true
        for clusterIndex in 0..<k {
            if clusterSizes[clusterIndex] > 0 {
                // (6) recompute centroid as the mean of all points assigned to it
                let newCentroid = clusterPointSums[clusterIndex] / Float(clusterSizes[clusterIndex])
                // (7) update the centroid if the new centroid not near the current centroid
                if simd_fast_distance(centroids[clusterIndex], newCentroid) > 0.0001 {
                    centroids[clusterIndex] = newCentroid
                    converged = false
                }
            }
        }
        
        if converged {
            print("converged at step \(iteration)")
            break
        }
    }
    
    return clusterIndices
}

private func parallel1(points: inout [Point2D], k: Int, maxIterations: Int) async -> [UInt32] {
    // (1) initially assign the centroids to the first k points
    var centroids:[Point2D] = []
    for i in 0..<k {
        centroids.append(points[i])
    }
    
    let processor = ParallelProcessor()
    
    // (2) create buffer for the points
    let pointsBuffer = processor.device.makeBuffer(bytesNoCopy: &points, length: MemoryLayout<Point2D>.stride * points.count, options: .storageModeShared)!

    // (3) create buffers for the centroids and the number of centroids
    let centroidsBuffer = processor.device.makeBuffer(bytesNoCopy: &centroids, length: MemoryLayout<Point2D>.stride * centroids.count, options: .storageModeShared)!
    var centroidsCountU32 = UInt32(k)
    let centroidsCountBuffer = processor.device.makeBuffer(bytes: &centroidsCountU32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    
    // (4) create a buffer that will hold each point's cluster index assignment
    var clusterIndices: [UInt32] = Array(repeating: 0, count: points.count)
    let clusterIndicesBuffer = processor.device.makeBuffer(bytesNoCopy: &clusterIndices, length: MemoryLayout<UInt32>.stride * points.count, options: .storageModeShared)!
        
    var options = ProcessOptions(threadsPerGrid: points.count, resetCommandAllocator: false)
    for iteration in 0..<maxIterations {
        // (5) determine cluster for each point on the GPU
        options.threadsPerGrid = points.count
        await processor.process(kernelName: "kMeansClusterAssignment", buffers: pointsBuffer, centroidsBuffer, centroidsCountBuffer, clusterIndicesBuffer, options: options)
        
        // (6) calculate cluster sizes and cluster point sums for each point on the CPU
        var pointSums: [Point2D] = Array(repeating: Point2D(0, 0), count: k)
        var clusterSizes: [UInt32] = Array(repeating: 0, count: k)
        for i in 0..<points.count {
            let clusterIndex = Int(clusterIndices[i])
            pointSums[clusterIndex] += points[i]
            clusterSizes[clusterIndex] += 1
        }
        
        // (7) calculate centroids for each cluster
        var converged = true
        for clusterIndex in 0..<k {
            if clusterSizes[clusterIndex] > 0 {
                // (8) recompute centroid as the mean of all points assigned to it
                let newCentroid = pointSums[clusterIndex]  / Float(clusterSizes[clusterIndex])
                // (9) update the centroid if the new centroid not near the current centroid
                if simd_fast_distance(centroids[clusterIndex], newCentroid) > 0.0001 {
                    centroids[clusterIndex] = newCentroid
                    converged = false
                }
            }
        }
        
        if converged {
            print("converged at step \(iteration)")
            break
        }
    }
    
    return clusterIndices
}

private func parallel2(points: inout [Point2D], k: Int, maxIterations: Int, m: Int) async -> [UInt32] {
    var centroids:[Point2D] = []
    for i in 0..<k {
        centroids.append(points[i])
    }
    
    let processor = ParallelProcessor()
    
    let pointsBuffer = processor.device.makeBuffer(bytesNoCopy: &points, length: MemoryLayout<Point2D>.stride * points.count, options: .storageModeShared)!
    var pointCountU32 = UInt32(points.count)
    let pointCountBuffer = processor.device.makeBuffer(bytes: &pointCountU32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    
    let centroidsBuffer = processor.device.makeBuffer(bytesNoCopy: &centroids, length: MemoryLayout<Point2D>.stride * centroids.count, options: .storageModeShared)!
    var centroidsCountU32 = UInt32(k)
    let centroidsCountBuffer = processor.device.makeBuffer(bytes: &centroidsCountU32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    
    var clusterIndices: [UInt32] = Array(repeating: 0, count: points.count)
    let clusterIndicesBuffer = processor.device.makeBuffer(bytesNoCopy: &clusterIndices, length: MemoryLayout<UInt32>.stride * points.count, options: .storageModeShared)!
    
    // (1) create a buffer to hold `m` point sum arrays
    var partialPointSums = [Point2D](repeating: Point2D(0, 0), count: k * m)
    let partialPointSumsBuffer = processor.device.makeBuffer(bytesNoCopy: &partialPointSums, length: MemoryLayout<Point2D>.stride * k * m, options: .storageModeShared)!
    
    // (2) create a buffer to hold `m` cluster size arrays
    var partialClusterSizes = [UInt32](repeating: 0, count: k * m)
    let partialClusterSizesBuffer = processor.device.makeBuffer(bytesNoCopy: &partialClusterSizes, length: MemoryLayout<UInt32>.stride * k * m, options: .storageModeShared)!
    
    var options = ProcessOptions(threadsPerGrid: points.count, resetCommandAllocator: false)
    for iteration in 0..<maxIterations {
        // (3) determine cluster for each point
        options.threadsPerGrid = points.count
        await processor.process(kernelName: "kMeansClusterAssignment", buffers: pointsBuffer, centroidsBuffer, centroidsCountBuffer, clusterIndicesBuffer, options: options)
        
        // (4) determine cluster sizes and cluster point sums partials
        options.threadsPerGrid = m
        await processor.process(kernelName: "kMeansClusterPointSumsClusterSizes", buffers: pointsBuffer, pointCountBuffer, centroidsCountBuffer, clusterIndicesBuffer, partialPointSumsBuffer, partialClusterSizesBuffer, options: options)

        // (5) sum cluster sizes and cluster point sums partials
        var pointSums: [Point2D] = Array(repeating: Point2D(0, 0), count: k)
        var clusterSizes: [UInt32] = Array(repeating: 0, count: k)
        for i in 0..<partialPointSums.count {
            let clusterIndex = Int(i) % k
            pointSums[clusterIndex] += partialPointSums[i]
            clusterSizes[clusterIndex] += partialClusterSizes[i]
        }
        
        var converged = true
        for clusterIndex in 0..<k {
            if clusterSizes[clusterIndex] > 0 {
                let newCentroid = pointSums[clusterIndex]  / Float(clusterSizes[clusterIndex])
                if simd_fast_distance(centroids[clusterIndex], newCentroid) > 0.0001 {
                    centroids[clusterIndex] = newCentroid
                    converged = false
                }
            }
        }
        
        if converged {
            print("converged at step \(iteration)")
            break
        }
    }
    
    return clusterIndices
}

nonisolated func kMeansClustering(points: inout [Point2D], k: Int) async -> ([UInt32], [UInt32]) {
    let maxIterations = 10
    
    var startTime = CFAbsoluteTimeGetCurrent()
    print("Parallel K-means 1 \(points.count), \(k)")
    let parallelClusterIndices1 = await parallel1(points: &points, k: k, maxIterations: maxIterations)
    print(String(format: "elapsed time: %.4f seconds", CFAbsoluteTimeGetCurrent() - startTime))

    let ms = [64, 128, 256, 512]
    var parallelClusterIndices2: [UInt32] = []
    for m in ms {
        startTime = CFAbsoluteTimeGetCurrent()
        print("\nParallel K-means 2 \(points.count), \(k), \(m)")
        parallelClusterIndices2 = await parallel2(points: &points, k: k, maxIterations: maxIterations, m: m)
        print(String(format: "elapsed time: %.4f seconds", CFAbsoluteTimeGetCurrent() - startTime))
    }

    startTime = CFAbsoluteTimeGetCurrent()
    print("\nSequential K-means \(points.count), \(k)")
    let sequentialClusterIndices = sequential(points: points, k: k, maxIterations: maxIterations)
    print(String(format: "Sequential elapsed time: %.4f seconds", CFAbsoluteTimeGetCurrent() - startTime))
    
    return (sequentialClusterIndices, parallelClusterIndices2)
}
