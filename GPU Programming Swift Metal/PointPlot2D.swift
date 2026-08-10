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
import SwiftUI
import simd

// MARK: - Model

struct PointSeries: Identifiable {
    enum Style { case scatter(radius: CGFloat = 3), line(width: CGFloat = 2) }
    let id = UUID()
    let name: String
    let points: [Point2D]
    let color: Color
    let style: Style
}

// MARK: - Plot

struct PointPlot2D: View {
    let series: [PointSeries]
    var showAxes: Bool = true
    var showGrid: Bool = true
    var padding: CGFloat = 24

    // Compute overall bounds across all series
    private var bounds: (minX: CGFloat, maxX: CGFloat, minY: CGFloat, maxY: CGFloat) {
        var minX = CGFloat.infinity, maxX = -CGFloat.infinity
        var minY = CGFloat.infinity, maxY = -CGFloat.infinity
        for s in series {
            for p in s.points {
                let x = CGFloat(p.x), y = CGFloat(p.y)
                minX = min(minX, x); maxX = max(maxX, x)
                minY = min(minY, y); maxY = max(maxY, y)
            }
        }
        // Handle empty or flat ranges
        if !minX.isFinite { minX = 0; maxX = 1 }
        if !minY.isFinite { minY = 0; maxY = 1 }
        if abs(maxX - minX) < .ulpOfOne { maxX = minX + 1 }
        if abs(maxY - minY) < .ulpOfOne { maxY = minY + 1 }
        return (minX, maxX, minY, maxY)
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Canvas { context, size in
                let b = bounds
                let plotRect = CGRect(
                    x: padding, y: padding,
                    width: max(1, size.width - 2 * padding),
                    height: max(1, size.height - 2 * padding)
                )

                // Helpers
                func mapX(_ x: CGFloat) -> CGFloat {
                    let t = (x - b.minX) / (b.maxX - b.minX)
                    return plotRect.minX + t * plotRect.width
                }
                func mapY(_ y: CGFloat) -> CGFloat {
                    let t = (y - b.minY) / (b.maxY - b.minY)
                    // flip Y for screen coords
                    return plotRect.maxY - t * plotRect.height
                }

                // Grid
                if showGrid {
                    var grid = Path()
                    let tickCount = 5
                    for i in 0...tickCount {
                        let tx = plotRect.minX + (CGFloat(i)/CGFloat(tickCount)) * plotRect.width
                        grid.move(to: CGPoint(x: tx, y: plotRect.minY))
                        grid.addLine(to: CGPoint(x: tx, y: plotRect.maxY))
                        let ty = plotRect.minY + (CGFloat(i)/CGFloat(tickCount)) * plotRect.height
                        grid.move(to: CGPoint(x: plotRect.minX, y: ty))
                        grid.addLine(to: CGPoint(x: plotRect.maxX, y: ty))
                    }
                    context.stroke(grid, with: .color(.secondary.opacity(0.25)), lineWidth: 0.5)
                }

                // Axes
                if showAxes {
                    var axes = Path()
                    axes.addRect(plotRect)
                    context.stroke(axes, with: .color(.secondary), lineWidth: 1)
                }

                // Series
                for s in series {
                    switch s.style {
                    case .scatter(let radius):
                        var dots = Path()
                        for p in s.points {
                            let x = mapX(CGFloat(p.x))
                            let y = mapY(CGFloat(p.y))
                            let r = radius
                            dots.addEllipse(in: CGRect(x: x - r, y: y - r, width: r, height: r))
                        }
                        context.fill(dots, with: .color(s.color))

                    case .line(let width):
                        guard s.points.count >= 2 else { continue }
                        var path = Path()
                        let first = s.points[0]
                        path.move(to: CGPoint(x: mapX(CGFloat(first.x)), y: mapY(CGFloat(first.y))))
                        for p in s.points.dropFirst() {
                            path.addLine(to: CGPoint(x: mapX(CGFloat(p.x)), y: mapY(CGFloat(p.y))))
                        }
                        context.stroke(path, with: .color(s.color), lineWidth: width)
                    }
                }
            }
            .background(Color(.windowBackgroundColor))

            // Legend
            HStack(spacing: 12) {
                ForEach(series) { s in
                    HStack(spacing: 6) {
                        Circle().fill(s.color).frame(width: 10, height: 10)
                        Text(s.name).font(.caption)
                    }
                }
            }
            .padding(.horizontal, 8)
        }
        .padding(.vertical, 4)
    }
}
