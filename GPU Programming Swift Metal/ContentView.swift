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

import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var mandlebrotStore: MandelbrotStore
    @EnvironmentObject private var kmeansParallelPointStore: KMeansParallelPlotStore
    @EnvironmentObject private var kmeansSequentialPointStore: KMeansSequentialPlotStore
    
    @Environment(\.openWindow) private var openWindow
    
    var body: some View {
        VStack(spacing: 50) {
            HStack(spacing: 50) {
                Button("Metal Device") {
                    Task {
                        metalDevice()
                    }
                }
                .padding()
                .background(.pink)
                .foregroundColor(.white)
                .clipShape(Capsule())
                
                Button("Metal Thread Dispatching") {
                    Task {
                        await metalThreadDispatching()
                    }
                }
                .padding()
                .background(.indigo)
                .foregroundColor(.white)
                .clipShape(Capsule())
                
                Button("Metal Memory") {
                    Task {
                        await metalMemory()
                    }
                }
                .padding()
                .background(.blue)
                .foregroundColor(.white)
                .clipShape(Capsule())

                Button("Metal Thread Coordination") {
                    Task {
                        await metalThreadCoordination()
                    }
                }
                .padding()
                .background(.orange)
                .foregroundColor(.white)
                .clipShape(Capsule())
            }
            
            HStack(spacing: 50) {
                Button("Mandelbrot") {
                    Task.detached(priority: .background) {
                        let size = 4_000
                        let pixels = await mandelbrot(width: size, height: size)
                        await MainActor.run {
                            mandlebrotStore.update(from: pixels, width: size, height: size)
                            openWindow(id: "mandelbrot-window")
                        }
                    }
                }
                .padding()
                .background(.pink)
                .foregroundColor(.white)
                .clipShape(Capsule())
                
                Button("K-Means Clustering") {
                    Task.detached(priority: .background) {
                        var minX:Int32 = -100
                        var maxX:Int32 = 100
                        var minY:Int32 = -100
                        var maxY:Int32 = 100
                        
                        var ns: [Int] = []
                        ns.append(Int(pow(2.0, 20.0)))
                        ns.append(Int(pow(2.0, 22.0)))
                        ns.append(Int(pow(2.0, 26.0)))
                        let ks = [10, 20, 30]
                        for n in ns {
                            for k in ks {
                                var points = await ParallelData.randomPoint2DArray(length: n, minX: &minX, maxX: &maxX, minY: &minY, maxY: &maxY)
                                let (sequentialClusterIndices, parallelClusterIndices) = await kMeansClustering(points: &points, k: k)
                                let sequentialSeries = pointSeriesFromPoints(points: points, k: k, clusterIndices: sequentialClusterIndices)
                                let parallelSeries = pointSeriesFromPoints(points: points, k: k, clusterIndices: parallelClusterIndices)
                                await MainActor.run {
                                    kmeansSequentialPointStore.latestSeries = sequentialSeries
                                    openWindow(id: "kmeans-sequential-window")
                                }
                                await MainActor.run {
                                    kmeansParallelPointStore.latestSeries = parallelSeries
                                    openWindow(id: "kmeans-parallel-window")
                                }
                            }
                        }
                    }
                }
                .padding()
                .background(.indigo)
                .foregroundColor(.white)
                .clipShape(Capsule())
                
                Button("Game of Life") {
                    Task {
                        await GameOfLifeController.shared.startSimulation()
                        openWindow(id: "GameOfLifeWindow")
                    }
                }
                .padding()
                .background(.blue)
                .foregroundColor(.white)
                .clipShape(Capsule())

                Button("Gray-Scott Model") {
                    Task {
                        await GrayScottController.shared.startSimulation()
                        openWindow(id: "GrayScottWindow")
                    }
                }
                .padding()
                .background(.orange)
                .foregroundColor(.white)
                .clipShape(Capsule())
            }
            
            HStack(spacing: 50) {
                Button("Array Max") {
                    Task.detached(priority: .background) {
                        await arrayMax()
                    }
                }
                .padding()
                .background(.pink)
                .foregroundColor(.white)
                .clipShape(Capsule())
                
                Button("Array Sum") {
                    Task.detached(priority: .background) {
                        await arraySum()
                    }
                }
                .padding()
                .background(.indigo)
                .foregroundColor(.white)
                .clipShape(Capsule())
                
                Button("Sorting") {
                    Task.detached(priority: .background) {
                        await metalSort()
                    }
                }
                .padding()
                .background(.blue)
                .foregroundColor(.white)
                .clipShape(Capsule())
                
                Button("Matrix Transpose") {
                    Task.detached(priority: .background) {
                        await matrixTranspose()
                    }
                }
                .padding()
                .background(.orange)
                .foregroundColor(.white)
                .clipShape(Capsule())
            }
            
            HStack(spacing: 50) {
                Button("Matrix Multiplication") {
                    Task.detached(priority: .background) {
                        await matrixMultiplication()
                    }
                }
                .padding()
                .background(.pink)
                .foregroundColor(.white)
                .clipShape(Capsule())
                
                Button("Systolic Arrays") {
                    Task {
                        await systolicArrays()
                    }
                }
                .padding()
                .background(.indigo)
                .foregroundColor(.white)
                .clipShape(Capsule())
                
                Button("Fast Fourier Transform") {
                    Task.detached(priority: .background) {
                        await fastFourierTransform()
                    }
                }
                .padding()
                .background(.blue)
                .foregroundColor(.white)
                .clipShape(Capsule())
                
                Button("Single Head Attention") {
                    Task.detached(priority: .background) {
                        await singleHeadAttention()
                    }
                }
                .padding()
                .background(.orange)
                .foregroundColor(.white)
                .clipShape(Capsule())
            }
            
            HStack(spacing: 50) {
                Button("Multi-Head Attention") {
                    Task.detached(priority: .background) {
                        await multiHeadAttention()
                    }
                }
                .padding()
                .background(.pink)
                .foregroundColor(.white)
                .clipShape(Capsule())

                Button("Multi-Head Attention Tensor") {
                    Task.detached(priority: .background) {
                        await multiHeadAttentionTensor()
                    }
                }
                .padding()
                .background(.indigo)
                .foregroundColor(.white)
                .clipShape(Capsule())
                
                Button("Profiling Basics") {
                    Task {
                        await metalProfiling()
                    }
                }
                .padding()
                .background(.blue)
                .foregroundColor(.white)
                .clipShape(Capsule())
                
                Button("For Loop Profiling") {
                    Task {
                        await forLoopProfiling()
                    }
                }
                .padding()
                .background(.orange)
                .foregroundColor(.white)
                .clipShape(Capsule())
            }
        }
        .padding()
    }
}

nonisolated func pointSeriesFromPoints(points: [Point2D], k: Int, clusterIndices: [UInt32]) -> [PointSeries] {
    var clusters = Array(repeating: [Point2D](), count: k)
    for i in 0..<points.count {
        let clusterIndex = Int(clusterIndices[i])
        clusters[clusterIndex].append(points[i])
    }
    
    var series: [PointSeries] = []
    for i in 0..<clusters.count {
        let s = PointSeries(name: String(format: "Cluster %d", i + 1),
                            points: clusters[i],
                            color: Color(hue: Double(i)/Double(k), saturation: 0.8, brightness: 0.8),
                            style: .scatter(radius: 2.0))
        series.append(s)
    }
    
    return series
}

#Preview {
    ContentView()
}
