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
import Metal
import SwiftUI

@main
struct GPUProgrammingSwiftMetalApp: App {
    @StateObject private var mandelbrotStore = MandelbrotStore()
    @StateObject private var kmeansParallelPointStore = KMeansParallelPlotStore()
    @StateObject private var kmeansSequentialPointStore = KMeansSequentialPlotStore()
    
    init() {
        
    }
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(kmeansParallelPointStore)
                .environmentObject(kmeansSequentialPointStore)
                .environmentObject(mandelbrotStore)
        }
        
        Window("Mandelbrot", id: "mandelbrot-window") {
            MandelbrotWindowView()
                .environmentObject(mandelbrotStore)
        }
        .windowResizability(.contentSize)
        .defaultLaunchBehavior(.suppressed)
        
        Window("K-means Parallel", id: "kmeans-parallel-window") {
            PointPlot2D(series: kmeansParallelPointStore.latestSeries,
                        showAxes: true, showGrid: true)
            .padding()
            .background(.background)
        }
        .windowResizability(.contentSize)
        .defaultLaunchBehavior(.suppressed)
        
        Window("K-means Sequential", id: "kmeans-sequential-window") {
            PointPlot2D(series: kmeansSequentialPointStore.latestSeries,
                        showAxes: true, showGrid: true)
            .padding()
            .background(.background)
        }
        .windowResizability(.contentSize)
        .defaultLaunchBehavior(.suppressed)
        
        Window("GameOfLife", id: "GameOfLifeWindow") {
            GameOfLifeWindowView()
        }
        .defaultSize(width: 1280, height: 720)
        .defaultLaunchBehavior(.suppressed)
        
        Window("GrayScott", id: "GrayScottWindow") {
            GrayScottWindowView().background(
                WindowSizeFixer(size: CGSize(width: 256, height: 290))
            )
        }
        .defaultSize(width: 256, height: 290)
        .defaultLaunchBehavior(.suppressed)

    }
}
