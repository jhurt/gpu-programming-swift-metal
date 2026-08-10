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

#include <metal_stdlib>
using namespace metal;

kernel void kMeansClusterAssignment(const device float2* points [[buffer(0)]],
                                    const device float2* centroids [[buffer(1)]],
                                    constant uint& k [[buffer(2)]],
                                    device uint* clusterIndices [[buffer(3)]],
                                    uint threadPositionInGrid [[thread_position_in_grid]]) {
    float minimumDistance = FLT_MAX;
    int closestCentroidIndex = -1;
    
    for (uint i = 0; i < k; i++) {
        float distance = fast::distance(points[threadPositionInGrid], centroids[i]);
        if (distance < minimumDistance) {
            minimumDistance = distance;
            closestCentroidIndex = i;
        }
    }
    
    clusterIndices[threadPositionInGrid] = closestCentroidIndex;
}

kernel void kMeansClusterPointSumsClusterSizes(const device float2* points [[buffer(0)]],
                                               constant uint& pointCount [[buffer(1)]],
                                               constant uint& k [[buffer(2)]],
                                               const device uint* clusterIndices [[buffer(3)]],
                                               device float2* partialPointSums [[buffer(4)]],
                                               device uint* partialClusterSizes [[buffer(5)]],
                                               uint threadsPerGrid [[threads_per_grid]],
                                               uint threadPositionInGrid [[thread_position_in_grid]]) {
    // (1) zero out the part of the partials this thread is responsible for
    for (uint clusterIndex = threadPositionInGrid * k; clusterIndex < threadPositionInGrid * k + k; clusterIndex++) {
        partialPointSums[clusterIndex] = float2(0.0, 0.0);
        partialClusterSizes[clusterIndex] = 0;
    }
    
    // (2) calculate cluster sizes and cluster point sums for the partials this thread is responsible for
    uint partitionCount = pointCount / threadsPerGrid;
    uint startPointIndex = threadPositionInGrid * partitionCount;
    uint endPointIndex = startPointIndex + partitionCount;
    endPointIndex = min(endPointIndex, pointCount);
    for (uint pointIndex = startPointIndex; pointIndex < endPointIndex; pointIndex++) {
        uint clusterIndex = clusterIndices[pointIndex];
        uint clusterPartitionIndex = threadPositionInGrid * k + clusterIndex;
        partialPointSums[clusterPartitionIndex] += points[pointIndex];
        partialClusterSizes[clusterPartitionIndex] += 1;
    }
}
